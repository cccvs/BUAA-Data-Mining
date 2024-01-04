import pandas as pd
import geopandas as gpd
import numpy as np

df = pd.read_csv("./data/mr.csv", sep=";")


def extract_cpath(cpath):
    print(type(cpath), cpath)
    if cpath == "":
        return []
    return [int(s) for s in cpath.split(",")]


df["cpath_list"] = df.apply(lambda row: extract_cpath(row.cpath), axis=1)
print(df)

all_edge_ids = np.unique(np.hstack(df.cpath_list)).tolist()

network_gdf = gpd.read_file("./data/road_split.shp")
network_gdf.id = network_gdf.id.astype(int)
network_gdf.head()

edges_df = network_gdf[network_gdf.id.isin(all_edge_ids)].reset_index()
edges_df["points"] = edges_df.apply(lambda row: len(row.geometry.coords), axis=1)
edges_df[["id", "source", "target", "geometry", "points"]].to_csv(
    "match.csv", sep=";", index=False
)
