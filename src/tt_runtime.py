from tt_simd_cluster import tt_simd_cluster
from tt_netlist import tt_netlist

class tt_runtime():
    def __init__(self,simd_cluster: tt_simd_cluster, netlist: tt_netlist, be_api, backend):
        self.simd_cluster = simd_cluster
        self.netlist = netlist
        self.be_api = be_api
        self.backend = backend
    def __del__(self):
        self.be_api.finish_child_process()
        self.backend.destroy()
