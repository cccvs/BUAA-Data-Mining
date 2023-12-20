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
ubodt_gen = UBODTGenAlgorithm(network, graph)
status = ubodt_gen.generate_ubodt("./data/ubodt.txt", 4, binary=False, use_omp=True)
print(status)
