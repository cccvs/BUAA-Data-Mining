import sys
sys.path.append('..')
import os
import math
import time
import random
import logging
import torch
import pickle
import pandas as pd
from ast import literal_eval
import numpy as np
import traj_dist.distance as tdist
import multiprocessing as mp
from functools import partial

from config import Config
from utils import tool_funcs
from utils.tool_funcs import lonlat2meters
from utils.edwp import edwp
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import seaborn as sns
import folium
from sklearn.manifold import MDS


def inrange(lon, lat):
    if lon <= Config.min_lon or lon >= Config.max_lon \
            or lat <= Config.min_lat or lat >= Config.max_lat:
        return False
    return True


def clean_and_output_data():
    _time = time.time()
    df = pd.read_csv(Config.root_dir + '/data/traj.csv')
    df.coordinates = df.coordinates.apply(literal_eval)  # convert string to list
    df = df.groupby('traj_id')['coordinates'].agg(list).reset_index()
    df.to_csv(Config.root_dir + '/data/beijing.csv', index=False)

    dfraw = pd.read_csv(Config.root_dir + '/data/beijing.csv')
    dfraw = dfraw.rename(columns = {"coordinates": "wgs_seq"})

    # length requirement
    dfraw.wgs_seq = dfraw.wgs_seq.apply(literal_eval)
    dfraw['trajlen'] = dfraw.wgs_seq.apply(lambda traj: len(traj))
    dfraw = dfraw[(dfraw.trajlen >= Config.min_traj_len) & (dfraw.trajlen <= Config.max_traj_len)]
    logging.info('Preprocessed-rm length. #traj={}'.format(dfraw.shape[0]))

    # range requirement
    dfraw['inrange'] = dfraw.wgs_seq.map(lambda traj: sum([inrange(p[0], p[1]) for p in traj]) == len(traj) ) # True: valid
    dfraw = dfraw[dfraw.inrange == True]
    logging.info('Preprocessed-rm range. #traj={}'.format(dfraw.shape[0]))

    # convert to Mercator
    dfraw['merc_seq'] = dfraw.wgs_seq.apply(lambda traj: [list(lonlat2meters(p[0], p[1])) for p in traj])

    logging.info('Preprocessed-output. #traj={}'.format(dfraw.shape[0]))
    dfraw = dfraw[['trajlen', 'wgs_seq', 'merc_seq']].reset_index(drop = True)

    dfraw.to_pickle(Config.dataset_file)
    logging.info('Preprocess end. @={:.0f}'.format(time.time() - _time))
    return


# ===calculate trajsimi distance matrix for trajsimi learning===
def traj_simi_computation(fn_name='hausdorff'):
    logging.info("traj_simi_computation starts. fn={}".format(fn_name))
    _time = time.time()

    # 1.
    trajs = pd.read_pickle(Config.dataset_file)
    trajs = _normalization([trajs])[0]

    logging.info("traj dataset size: {}".format(trajs.shape[0]))

    # 2.
    fn = _get_simi_fn(fn_name)
    simi_matrix = _simi_matrix(fn, trajs)  # [ [simi, simi, ... ], ... ]
    simi_matrix = np.triu(simi_matrix) + np.triu(simi_matrix, 1).T
    # 3.
    _output_file = '{}_traj_simi_dict_{}.pkl'.format(Config.dataset_file, fn_name)
    with open(_output_file, 'wb') as fh:
        tup = simi_matrix
        pickle.dump(tup, fh, protocol=pickle.HIGHEST_PROTOCOL)

    simi_df = pd.DataFrame(simi_matrix)
    # Save DataFrame to Excel file
    excel_output_file = '{}_traj_simi_{}.xlsx'.format(Config.dataset_file, fn_name)
    simi_df.to_excel(excel_output_file, index=False)
    return


def traj_clustering():
    trajs = pd.read_pickle(Config.dataset_file)
    trajs = _normalization([trajs])[0]
    simi_matrix = pd.read_pickle('{}_traj_simi_dict_edwp.pkl'.format(Config.dataset_file))
    n_cluster = 10  # 尝试不同的簇数量
    spectral_clustering = SpectralClustering(n_clusters=n_cluster, affinity='nearest_neighbors', n_neighbors=50, random_state=42)
    clusters = spectral_clustering.fit_predict(simi_matrix)
    _output_file = '{}_traj_cluster.pkl'.format(Config.dataset_file)
    with open(_output_file, 'wb') as fh:
        tup = clusters
        pickle.dump(tup, fh, protocol=pickle.HIGHEST_PROTOCOL)
    cluster_df = pd.DataFrame(clusters)
    # Save DataFrame to Excel file
    excel_output_file = '{}_traj_cluster.xlsx'.format(Config.dataset_file)
    cluster_df.to_excel(excel_output_file, index=False)
    return trajs, simi_matrix, clusters


def _normalization(lst_df):
    # lst_df: [df, df, df]
    xs = []
    ys = []
    for df in lst_df:
        for _, v in df.merc_seq.iteritems():
            arr = np.array(v)
            xs.append(arr[:, 0])
            ys.append(arr[:, 1])

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    mean = np.array([xs.mean(), ys.mean()])
    std = np.array([xs.std(), ys.std()])

    for i in range(len(lst_df)):
        lst_df[i].merc_seq = lst_df[i].merc_seq.apply(lambda lst: ((np.array(lst) - mean) / std).tolist())

    return lst_df


def _get_simi_fn(fn_name):
    fn = {'lcss': tdist.lcss, 'edr': tdist.edr, 'frechet': tdist.frechet,
          'discret_frechet': tdist.discret_frechet,
          'hausdorff': tdist.hausdorff, 'edwp': edwp}.get(fn_name, None)
    if fn_name == 'lcss' or fn_name == 'edr':
        fn = partial(fn, eps=Config.test_exp1_lcss_edr_epsilon)
    return fn


def _simi_matrix(fn, df):
    _time = time.time()

    l = df.shape[0]
    batch_size = 50
    # assert l % batch_size == 0 # todo

    # parallel init
    tasks = []
    for i in range(math.ceil(l / batch_size)):
        if i < math.ceil(l / batch_size) - 1:
            tasks.append((fn, df, list(range(batch_size * i, batch_size * (i + 1)))))
        else:
            tasks.append((fn, df, list(range(batch_size * i, l))))

    num_cores = int(mp.cpu_count()) - 6
    assert num_cores > 0
    logging.info("pool.size={}".format(num_cores))
    pool = mp.Pool(num_cores)
    lst_simi = pool.starmap(_simi_comp_operator, tasks)
    pool.close()

    # extend lst_simi to matrix simi and pad 0s
    lst_simi = sum(lst_simi, [])
    for i, row_simi in enumerate(lst_simi):
        lst_simi[i] = [0] * (i + 1) + row_simi
    assert sum(map(len, lst_simi)) == l ** 2
    logging.info('simi_matrix computation done., @={}, #={}'.format(time.time() - _time, len(lst_simi)))

    return lst_simi


def plot_similarity_matrix_heatmap(trajs, simi_matrix, clusters, num_trajectories=10):
    random_indices = np.random.choice(len(simi_matrix), num_trajectories, replace=False)
    selected_simi_matrix = [[simi_matrix[i][j] for j in random_indices] for i in random_indices]
    selected_clusters = clusters[random_indices]

    # 创建热图
    plt.figure(figsize=(14, 8))
    sns.heatmap(selected_simi_matrix, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=random_indices, yticklabels=random_indices)
    plt.title("Distance Matrix by EWpD")
    plt.savefig(f'cluster_heatmap.png')
    plt.close()

    # 创建 cluster 列表 DataFrame
    cluster_df = pd.DataFrame(columns=["Trajectory", "Cluster"])
    for i, cluster in zip(random_indices, clusters[random_indices]):
        cluster_df = cluster_df.append({"Trajectory": i, "Cluster": cluster}, ignore_index=True)

    # 输出 cluster 列表
    plt.figure(figsize=(6, 8))
    plt.axis("off")
    plt.table(cellText=cluster_df.values, colLabels=cluster_df.columns, cellLoc='center', loc='center')
    plt.title("Cluster Information")
    plt.savefig(f'cluster_table.png')
    plt.close()

    mds = MDS(n_components=2, dissimilarity='euclidean', random_state=42)
    embedded_points = mds.fit_transform(simi_matrix)

    # 创建散点图
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=embedded_points[:, 0], y=embedded_points[:, 1],
                    hue=clusters, palette="viridis", s=50, alpha=0.7)
    # 设置图形属性
    plt.title('Trajectory Clustering Scatter Plot with Distance Matrix (MDS)')
    plt.savefig(f'cluster_scatter.png')
    plt.close()

#     # 创建轨迹图
#     trajs = trajs.wgs_seq.tolist()
#     m = folium.Map(location=[trajs[0][0][1], trajs[0][0][0]], zoom_start=13)
#     index = 0
#     print(np.unique(selected_clusters))
#     for i, cluster in zip(random_indices, selected_clusters):
#         color = sns.color_palette("husl", n_colors=20)[cluster]
#         folium.PolyLine(trajs[i], color=color, weight=10, opacity=1).add_to(m)
#     m.save(f'trajectories_with_clusters.html')


# async operator
def _simi_comp_operator(fn, df_trajs, sub_idx):
    simi = []
    l = df_trajs.shape[0]
    for _i in sub_idx:
        t_i = np.array(df_trajs.iloc[_i].merc_seq)
        simi_row = []
        for _j in range(_i + 1, l):
            t_j = np.array(df_trajs.iloc[_j].merc_seq)
            simi_row.append(float(fn(t_i, t_j)))
        simi.append(simi_row)
    logging.debug('simi_comp_operator ends. sub_idx=[{}:{}], pid={}' \
                  .format(sub_idx[0], sub_idx[-1], os.getpid()))
    return simi


# nohup python ./preprocessing_porto.py &> ../result &
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format="[%(filename)s:%(lineno)s %(funcName)s()] -> %(message)s",
                        handlers=[
                            logging.FileHandler(Config.root_dir + '/exp/log/' + tool_funcs.log_file_name(), mode='w'),
                            logging.StreamHandler()]
                        )
    Config.dataset = 'beijing'
    Config.post_value_updates()

    clean_and_output_data()
    traj_simi_computation('edwp')  # edr edwp discret_frechet hausdorff
    trajs, simi_matrix, clusters = traj_clustering()
    silhouette_avg = silhouette_score(simi_matrix, clusters)
    print(f"Silhouette Score: {silhouette_avg}")
    ch_score = calinski_harabasz_score(simi_matrix, clusters)
    print(f"Calinski-Harabasz Index: {ch_score}")
    db_index = davies_bouldin_score(simi_matrix, clusters)
    print(f"Davies-Bouldin Index: {db_index}")
    plot_similarity_matrix_heatmap(trajs, simi_matrix, clusters)
