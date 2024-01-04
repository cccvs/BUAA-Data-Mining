from fmm import Network, NetworkGraph, STMATCH, STMATCHConfig
from fmm import GPSConfig, ResultConfig


network = Network("./data/road_split.shp")
print("Nodes {} edges {}".format(network.get_node_count(), network.get_edge_count()))
graph = NetworkGraph(network)

model = STMATCH(network, graph)

k = 8
k = 8
gps_error = 0.5
radius = 0.4
vmax = 30
factor = 1.5
stmatch_config = STMATCHConfig(k, radius, gps_error, vmax, factor)


# Define input data configuration
input_config = GPSConfig()
input_config.file = "./data/traj_new.csv"
input_config.id = "id"
input_config.gps_point = True
result_config = ResultConfig()
result_config.file = "./data/mr_stmatch.csv"
# result_config.output_config.write_opath = True
# result_config.output_config.write_cpath = True
# result_config.output_config.write_mgeom = True
# result_config.output_config.write_duration = True
# result_config.output_config.write_spdist = True
# result_config.output_config.write_length = True
result_config.output_config.write_all = True


print(result_config.to_string())
status = model.match_gps_file(input_config, result_config, stmatch_config)
print(status)
