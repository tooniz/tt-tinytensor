import logging
from tt_malloc import tt_malloc
#from tt_netlist import tt_netlist
from tt_dtype import tt_dtype
from tt_dtype import tt_op_dtype
from tt_dtype import tt_math_fidelity
from tt_dtype import block_size_bytes
import torch
from enum import Enum
import eager_backend.backend_api as be_api
from test_utils import py_desc, py_tensor
from eager_backend import DataFormat, BackendType, BackendDevice, BackendStatusCode, IOType, IOLocation
from eager_backend.backend_api import Backend, BackendConfig, PytorchTensorDesc

class tt_dram_chan_picker(Enum):
    constant = 0
    distributed = 1
    roundrobin_block = 2
    roundrobin_tensor = 3

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
        self.be_api.push_tensor(weight['loc'], chip_id, py_desc(weight['chan']), py_desc(weight['addr']), py_desc(weight['data']), IOType.RandomAccess, 0)

    def read_tensor_slice(self,chip_id,data,chan,addr,torch_dtype):
        weight = {
            'data': data,
            'chan': chan,
            'addr': addr,
            'loc': IOLocation.Dram
        }
        assert weight['data'].dtype == torch_dtype
        weight_desc = py_desc(weight['data'])
        self.be_api.get_tensor(weight['loc'], chip_id, py_desc(weight['chan']), py_desc(weight['addr']), weight_desc, IOType.RandomAccess, 0, False)
        out = py_tensor(weight_desc)
        return out

class tt_simd_cluster():
    def __init__(self, r: int, c: int, ids: tuple, be_api = None, netlist=None, arch=BackendDevice.Grayskull):
        self.r = r
        self.c = c
        self.r_cores = 10
        self.c_cores = 12 if arch == BackendDevice.Grayskull else 8
        self.queue_lim = 30
        self.ids = ids
        self.allocators = {}
        self.be_api = be_api
        self.arch = arch
        self.netlist = netlist

        if (self.arch == BackendDevice.Grayskull):
            self.chan_picker = tt_dram_chan_picker.distributed
        else:
            # wh a0 enforces dram buffers to be on the same chan for scatter, Sean has a fix WIP
            self.chan_picker = tt_dram_chan_picker.roundrobin_tensor

        #self.netlist_api = tt_netlist
        if(be_api is not None):
            self.dram_accessor = tt_dram_accessor(be_api)

    def get_chip_id(self, r: int, c: int):
        id = r * self.c + c
        return id

    def write_tensor_slice_to_dram(self, chip_id, data, chan, address):
        self.dram_accessor.write_tensor_slice(chip_id=chip_id, data=data, chan=chan, addr=address)

    def read_tensor_slice_from_dram(self, chip_id, data_shape, chan, address, torch_dtype):
        return self.dram_accessor.read_tensor_slice(chip_id, data_shape, chan, address, torch_dtype)

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

    def num_dram_channels(self):
        if (self.arch == BackendDevice.Grayskull):
            return 8
        elif (self.arch == BackendDevice.Wormhole or self.arch == BackendDevice.Wormhole_B0):
            return 6
        else:
            logging.exception(f"Unsupported architecture {self.arch}")
            assert False

    def check_allocator_end_state(self):
        # Go through allocators and check that everything was de-allocated properly
        for allocator in self.allocators.values():
            # check the free list is back to being fully free
            assert list(allocator.free_block_index_tensor.shape)[0] == allocator.num_blocks, "Error: deallocator did not put back all blocks"

            # check the blocks in the free list are unique
            unique = torch.unique(allocator.free_block_index_tensor)
            assert unique.shape == allocator.free_block_index_tensor.shape, "Error: the free list got poluted, with duplicate blocks"
