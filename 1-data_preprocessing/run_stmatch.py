from fmm import Network, NetworkGraph, STMATCH, STMATCHConfig
from fmm import GPSConfig, ResultConfig


network = Network("./data/road_new.shp")
print("Nodes {} edges {}".format(network.get_node_count(), network.get_edge_count()))
graph = NetworkGraph(network)

model = STMATCH(network, graph)

k = 4
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
result_config.file = "./data/mr.csv"
result_config.output_config.write_opath = True
print(result_config.to_string())
status = model.match_gps_file(input_config, result_config, stmatch_config)
print(status)
