import logging
from tt_malloc import tt_malloc
from enum import Enum
import eager_backend.backend_api as be_api
from test_utils import py_desc, py_tensor
from eager_backend import DataFormat, BackendType, BackendDevice, BackendStatusCode, IOType, IOLocation
from eager_backend.backend_api import Backend, BackendConfig, PytorchTensorDesc

class tt_dtype(Enum):
    Bfp2_b = 1
    Bfp4_b = 2
    Bfp8_b = 3
    Float16_a = 4
    Float16_b = 5
    Float32 = 6

class tt_math_fidelity(Enum):
    LoFi = 1
    HiFi = 2

class tt_op_dtype():
    def __init__(self, dtype, dype_intermed=tt_dtype.Float16_b, dtype_accum=tt_dtype.Float16_b, fidelity: tt_math_fidelity = tt_math_fidelity.HiFi):
        self.dt = dtype
        self.dt_int = dype_intermed
        self.dt_acc = dtype_accum
        self.fidelity = fidelity

def block_size_bytes(dtype, block_size, debug=False):
    tile_size_dict = {}
    if debug is False:
        tile_size_dict[tt_dtype.Bfp2_b] = 352
        tile_size_dict[tt_dtype.Bfp4_b] = 608
        tile_size_dict[tt_dtype.Bfp8_b] = 1120
        tile_size_dict[tt_dtype.Float16_a] = 2080
        tile_size_dict[tt_dtype.Float16_b] = 2080
        tile_size_dict[tt_dtype.Float32] = 4128
    else:
        tile_size_dict[tt_dtype.Bfp2_b] = 352 * 2
        tile_size_dict[tt_dtype.Bfp4_b] = 608 * 2
        tile_size_dict[tt_dtype.Bfp8_b] = 1120 * 2
        tile_size_dict[tt_dtype.Float16_a] = 2080 * 2
        tile_size_dict[tt_dtype.Float16_b] = 2080 * 2
        tile_size_dict[tt_dtype.Float32] = 4128 * 2

    return int((block_size/32)*(block_size/32)*tile_size_dict[dtype])

class tt_dram_accessor():
    def __init__(self, be_api):
        self.be_api = be_api
    def write_tensor_slice(self,chip_id,data,chan,addr):
        weight = {
            'data': data,
            'chan': chan,
            'addr': addr,
            'loc': IOLocation.Dram
        }
        self.be_api.init_queue(weight['loc'], chip_id, py_desc(weight['chan']), py_desc(weight['addr']), 1)
        self.be_api.push_tensor(IOLocation.Dram, chip_id, py_desc(weight['chan']), py_desc(weight['addr']), py_desc(weight['data']), IOType.RandomAccess, 0)
    def read_tensor_slice(self,chip_id,data,chan,addr):
        weight = {
            'data': data,
            'chan': chan,
            'addr': addr,
            'loc': IOLocation.Dram
        }
        weight_desc = py_desc(weight['data'])
        self.be_api.get_tensor(weight['loc'], chip_id, py_desc(weight['chan']), py_desc(weight['addr']), weight_desc, IOType.RandomAccess, 0, False)
        out = py_tensor(weight_desc)
        return out

class tt_simd_cluster():
    def __init__(self, r: int, c: int, ids: tuple, be_api = None):
        self.r = r
        self.c = c
        self.ids = ids
        self.allocators = {}
        if(be_api is not None):
            self.dram_accessor = tt_dram_accessor(be_api)

    def compile_netlist(self):
        pass

    def run_netlist(self):
        pass

    def __del__(self):
        pass

    def write_tensor_slice_to_dram(self, chip_id, data, chan, address):
        self.dram_accessor.write_tensor_slice(chip_id=chip_id, data=data, chan=chan, addr=address)

    def read_tensor_slice_from_dram(self, chip_id, data_shape, chan, address):
        return self.dram_accessor.read_tensor_slice(chip_id, data_shape, chan, address)

    def set_up_allocators(self, alloc_list: list): # list of 4 entry tuples (dtype, block size, number of blocks, base_address)
        for alloc_data in alloc_list:
            self.allocators[(alloc_data[1],alloc_data[0])] = tt_malloc(block_size_bytes(dtype=alloc_data[0],block_size=alloc_data[1]), alloc_data[2], alloc_data[3])

    def allocate_tensor(self, tensor):
        if (tensor.block_size,tensor.dtype) in self.allocators:
            return self.allocators[(tensor.block_size,tensor.dtype)].allocate_tensor(tensor.address_tensor)
        else:
            logging.exception("Trying to allocate dram without having properly configured allocator for a given block, data type")
            assert False

    def deallocate_tensor(self, tensor):
        if (tensor.block_size,tensor.dtype) in self.allocators:
            self.allocators[(tensor.block_size,tensor.dtype)].deallocate_tensor(tensor.address_tensor)
        else:
            logging.exception("Trying to de-allocate dram without having properly configured allocator for a given block, data type")
            assert False

