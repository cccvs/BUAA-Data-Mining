from traclus import traclus as tr
import numpy as np
import pandas as pd
from ast import literal_eval


def clean_and_output_data():
    dfraw = pd.read_csv('traj.csv')
    dfraw.coordinates = dfraw.coordinates.apply(literal_eval)  # convert string to list
    trajectories = dfraw.groupby('traj_id')['coordinates'].agg(list).reset_index()
    trajectories.coordinates = trajectories.coordinates.apply(lambda traj_list: np.array(traj_list))
    trajectories.to_csv('trajectories.csv', index=False)
    return trajectories.coordinates.tolist()


def train():
    trajectories = clean_and_output_data()
    partitions, segments, dist_matrix, clusters, cluster_assignments, representative_trajectories = tr.traclus(
        trajectories)
    output_file_path = 'traclus_results.txt'

    with open(output_file_path, 'w') as f:
        f.write("Partitions:\n")
        f.write(str(partitions) + "\n\n")

        f.write("Segments:\n")
        f.write(str(segments) + "\n\n")

        f.write("Distance Matrix:\n")
        f.write(str(dist_matrix) + "\n\n")

        f.write("Clusters:\n")
        f.write(str(clusters) + "\n\n")

        f.write("Cluster Assignments:\n")
        f.write(str(cluster_assignments) + "\n\n")

        f.write("Representative Trajectories:\n")
        f.write(str(representative_trajectories) + "\n\n")

    print(f"Results have been written to {output_file_path}")
    return partitions, segments, dist_matrix, clusters, cluster_assignments, representative_trajectories


train()