import pandas as pd
from shapely.geometry import Point, LineString
from geopy.distance import geodesic
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
coord_list, coord_set, coord_dict = list(), set(), dict()
for i, row in df_road_old.iterrows():
    for x, y in zip(row["geometry"].xy[0], row["geometry"].xy[1]):
        if not (x, y) in coord_set:
            coord_list.append((x, y))
            coord_set.add((x, y))
coord_dict = {coord: i for i, coord in enumerate(coord_list)}
df_node = pd.DataFrame(coord_dict.items(), columns=["coordinates", "id"])
df_node["WKT"] = df_node["coordinates"].apply(lambda t: Point(t).wkt)
df_node.drop(["coordinates"], axis=1, inplace=True)
df_node.to_csv("./data/node_split.csv", index=False)

# Construct new roads
series_list = []
max_error_rate = 0.0
for i, row in df_road_old.iterrows():
    point_list = list(zip(*row["geometry"].xy))
    total_dist = 0.0
    for j in range(len(point_list) - 1):
        row_new = row.copy()
        row_new["geometry"] = LineString([point_list[j], point_list[j + 1]])
        row_new["length"] = geodesic(
            (point_list[j][1], point_list[j][0]),
            (point_list[j + 1][1], point_list[j + 1][0]),
        ).meters
        series_list.append(row_new)
        total_dist += row_new["length"]
    error_rate = abs(total_dist - row["length"]) / min(row["length"], total_dist)
    max_error_rate = max(max_error_rate, error_rate)
    print(f"Road {i}, distance error: {total_dist - row['length']}, error rate: {error_rate * 100}%")
print(f"Max error rate: {max_error_rate * 100}%")
df_road_split = pd.concat(series_list, axis=1, ignore_index=True).T
df_road_split.reset_index(drop=False, inplace=True)
df_road_split.rename(columns={"index": "id"}, inplace=True)
df_road_split["_uid_"] = df_road_split["id"]
df_road_split["WKT"] = df_road_split["geometry"].apply(lambda t: t.wkt)
df_road_split["source"] = df_road_split["geometry"].apply(
    lambda t: coord_dict[(t.xy[0][0], t.xy[1][0])]
)
df_road_split["target"] = df_road_split["geometry"].apply(
    lambda t: coord_dict[(t.xy[0][-1], t.xy[1][-1])]
)
df_road_split["cost"] = df_road_split["length"]
df_road_split["x1"] = df_road_split["geometry"].apply(lambda t: t.xy[0][0])
df_road_split["y1"] = df_road_split["geometry"].apply(lambda t: t.xy[1][0])
df_road_split["x2"] = df_road_split["geometry"].apply(lambda t: t.xy[0][-1])
df_road_split["y2"] = df_road_split["geometry"].apply(lambda t: t.xy[1][-1])
df_road_split.drop(["highway", "length", "lanes", "tunnel", "bridge", "maxspeed", "width", "alley", "roundabout"], axis=1, inplace=True)
df_road_split.to_csv("./data/road_split.csv", index=False)


# Convert traj to new csv
def str_to_time(time_str: str):
    dt = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%SZ")
    return int(dt.replace(tzinfo=pytz.UTC).timestamp())


df_traj = pd.read_csv("./data/traj.csv")
df_traj["x"] = df_traj["coordinates"].apply(lambda t: ast.literal_eval(t)[0])
df_traj["y"] = df_traj["coordinates"].apply(lambda t: ast.literal_eval(t)[1])
df_traj["timestamp"] = df_traj["time"].apply(str_to_time)
df_traj["id"] = df_traj["traj_id"]
df_traj.drop(["time", "traj_id", "coordinates", "current_dis", "speeds", "holidays"], axis=1, inplace=True)
df_traj.to_csv("./data/traj_new.csv", sep=";", index=False, index_label="id")
