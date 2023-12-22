from fmm import (
    Network,
    NetworkGraph,
    FastMapMatch,
    FastMapMatchConfig,
    UBODT,
    UBODTGenAlgorithm,
)

network = Network("./data/road_new.shp")
print("Nodes {} edges {}".format(network.get_node_count(), network.get_edge_count()))
graph = NetworkGraph(network)
# ubodt_gen = UBODTGenAlgorithm(network, graph)
# status = ubodt_gen.generate_ubodt("./data/ubodt.txt", delta=4, binary=False, use_omp=True)
# print(status)

from fmm import GPSConfig, ResultConfig
from fmm import (
    Network,
    NetworkGraph,
    FastMapMatch,
    FastMapMatchConfig,
    UBODT,
    UBODTGenAlgorithm,
)


network = Network("./data/road_new.shp")
graph = NetworkGraph(network)
ubodt = UBODT.read_ubodt_csv("./data/ubodt.txt")
model = FastMapMatch(network, graph, ubodt)
k = 4
radius = 0.4
gps_error = 0.5
fmm_config = FastMapMatchConfig(k, radius, gps_error)

# GPS config
input_config = GPSConfig()
input_config.file = "./data/traj_new.csv"
input_config.id = "id"
input_config.gps_point = True
# Result config
result_config = ResultConfig()
result_config.file = "./data/mr.csv"
result_config.output_config.write_opath = True
result_config.output_config.write_pgeom = True
result_config.output_config.write_offset = True
result_config.output_config.write_error = True
result_config.output_config.write_spdist = True
result_config.output_config.write_cpath = True
result_config.output_config.write_tpath = True
result_config.output_config.write_mgeom = True
result_config.output_config.write_ep = True
result_config.output_config.write_tp = True
result_config.output_config.write_length = True
result_config.output_config.write_duration = True
result_config.output_config.write_speed = True
print(result_config.to_string())
status = model.match_gps_file(input_config, result_config, fmm_config)
print(status)