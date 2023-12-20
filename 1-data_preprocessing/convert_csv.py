import pandas as pd
from shapely.geometry import Point, LineString
from datetime import datetime
import pytz
import ast

# Convert csv to data frame
df_road_old = pd.read_csv("./data/road.csv")
df_road_old["geometry"] = df_road_old["coordinates"].apply(
    lambda t: LineString(ast.literal_eval(t))
)
df_road_old.drop(["coordinates", "id"], axis=1, inplace=True)

# Construct nodes
coord_set, coord_dict = set(), dict()
for i, row in df_road_old.iterrows():
    for x, y in zip(row["geometry"].xy[0], row["geometry"].xy[1]):
        coord_set.add((x, y))
coord_dict = {coord: i for i, coord in enumerate(coord_set)}
df_node = pd.DataFrame(coord_dict.items(), columns=["coordinates", "id"])
df_node["WKT"] = df_node["coordinates"].apply(lambda t: Point(t).wkt)
df_node.drop(["coordinates"], axis=1, inplace=True)
df_node.to_csv("./data/node_new.csv", index=False)

# Construct new roads
series_list = []
for i, row in df_road_old.iterrows():
    point_list = list(zip(*row["geometry"].xy))
    for j in range(len(point_list) - 1):
        row_new = row.copy()
        row_new["geometry"] = LineString([point_list[j], point_list[j + 1]])
        series_list.append(row_new)
df_road_new = pd.concat(series_list, axis=1).T
df_road_new.reset_index(drop=True, inplace=True)
df_road_new["WKT"] = df_road_new["geometry"].apply(lambda t: t.wkt)
df_road_new["source"] = df_road_new["geometry"].apply(
    lambda t: coord_dict[(t.xy[0][0], t.xy[1][0])]
)
df_road_new["target"] = df_road_new["geometry"].apply(
    lambda t: coord_dict[(t.xy[0][-1], t.xy[1][-1])]
)
df_road_new.drop(["geometry"], axis=1, inplace=True)
df_road_new.to_csv("./data/road_new.csv", index=True, index_label="id")


# Convert traj to new csv
def str_to_time(time_str: str):
    dt = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%SZ")
    return int(dt.replace(tzinfo=pytz.UTC).timestamp())


df_traj = pd.read_csv("./data/traj.csv")
df_traj["x"] = df_traj["coordinates"].apply(lambda t: ast.literal_eval(t)[0])
df_traj["y"] = df_traj["coordinates"].apply(lambda t: ast.literal_eval(t)[1])
df_traj["timestamp"] = df_traj["time"].apply(str_to_time)
df_traj["id"] = df_traj["traj_id"]
df_traj.to_csv("./data/traj_new.csv", sep=";", index=False, index_label="id")
