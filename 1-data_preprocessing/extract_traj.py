import pandas as pd
import geopandas as gpd
import numpy as np

df = pd.read_csv("./data/mr_stmatch.csv", sep=";")

df.drop(columns=["opath", "cpath"], inplace=True)
df.rename(columns={"mgeom": "WKT"}, inplace=True)
df.to_csv("./data/match_traj.csv", index=False)