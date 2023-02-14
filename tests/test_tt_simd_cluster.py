import torch
import random
import logging
from tt_simd_cluster import tt_simd_cluster
from tt_simd_cluster import tt_dtype, tt_op_dtype
from tt_tensor import tt_tensor
import time
import torch
from tt_netlist import tt_netlist
from tt_netlist import tt_net_op_types
from tt_runtime import tt_runtime
import tt_functional as ttf
import eager_backend.backend_api as be_api
from test_utils import py_desc, py_tensor
from eager_backend import DataFormat, BackendType, BackendDevice, BackendStatusCode, IOType, IOLocation
from eager_backend.backend_api import Backend, BackendConfig, PytorchTensorDesc
from IPython import embed

def duplicate_alloc_test():
    logging.info("Testing that multiple calls to set up allocator with same setting results in one allocator being set up!")

    simd0  = tt_simd_cluster(2,2,(0,1,2,3))
    num_alloc_blocks = 10

    simd0.set_up_allocators([(tt_dtype.Bfp8_b,64,num_alloc_blocks,0)])
    simd0.set_up_allocators([(tt_dtype.Bfp8_b,64,num_alloc_blocks,0)])
    simd0.set_up_allocators([(tt_dtype.Bfp8_b,64,num_alloc_blocks,0)])

    simd0.set_up_allocators([(tt_dtype.Bfp8_b,32,num_alloc_blocks,0)])
    simd0.set_up_allocators([(tt_dtype.Bfp8_b,32,num_alloc_blocks,0)])
    simd0.set_up_allocators([(tt_dtype.Bfp8_b,32,num_alloc_blocks,0)])

    simd0.set_up_allocators([(tt_dtype.Float16_b,64,num_alloc_blocks,0)])
    simd0.set_up_allocators([(tt_dtype.Float16_b,64,num_alloc_blocks,0)])
    simd0.set_up_allocators([(tt_dtype.Float16_b,64,num_alloc_blocks,0)])

    # check that the above all resulted in a single allocators dictionary entry in simd cluster
    assert len(simd0.allocators) == 3

def grayskull_read_write_test():
    import time
    import torch

    target_arch = BackendDevice.Grayskull
    target_devices = {0}
    config = be_api.get_runtime_config(target_arch)
    backend = Backend(config, target_devices)
    be_api.initialize_child_process(target_arch, target_devices)

    simd0 = tt_simd_cluster(4,8, list(range(4*8)), be_api)
    simd0.set_up_allocators([(tt_dtype.Float32, 128, 10000, 0x20000000)])
    simd0.set_up_allocators([(tt_dtype.Float32, 256, 10000, 0x30000000)])

    for i in range(8):
        dims = random.choice([1,3,4])
        block_size = random.choice([128,256])
        tensor_size = random.choice([512])
        tens = torch.randn((1,1,dims,tensor_size,tensor_size))
        tt_tens = tt_tensor(block_size=block_size, simd_cluster=simd0, torch_tensor=tens, dtype=tt_dtype.Float32)
        tt_tens.to_device(0,tens)
        backend.wait_for_idle()
        ble = tt_tens.from_device(0)
        backend.wait_for_idle()
        diff = torch.isclose(tens,ble)
        inv = torch.logical_not(diff)
        indices = inv.nonzero()
        del(tt_tens)
        #IPython.embed()
        assert torch.allclose(tens,ble)

    check_allocator_end_state(simd0)
    simd0.be_api.finish_child_process()
    backend.destroy()
    logging.info("Passed grayskull tensor read/write test")

def grayskull_matmul_test():
    target_arch = BackendDevice.Grayskull
    target_devices = {0}
    config = be_api.get_runtime_config(target_arch)
    backend = Backend(config, target_devices)
    be_api.initialize_child_process(target_arch, target_devices)

    simd0 = tt_simd_cluster(4,8, list(range(4*8)), be_api)
    netlist = tt_netlist()

    # Run some matmuls
    random_slice_matmuls(10,simd0,netlist,backend)

    simd0.check_allocator_end_state()
    simd0.be_api.finish_child_process()
    backend.destroy()

    logging.info("Passed grayskull matmul test")

def random_slice_matmuls(count: int, simd0: tt_simd_cluster, netlist: tt_netlist, backend):
    for i in range(count):
        block_size_max_mul = random.choice([(64,5),(128,2),(256,1)]) # block size 512 overflows SRAM ,(512,1)])
        block_size = block_size_max_mul[0]
        block_mul  = random.randint(1,block_size_max_mul[1])
        id_mul = random.randint(1,2)
        number_of_inputs = 2
        number_of_dims = 3
        simd0.set_up_allocators([(tt_dtype.Float32, block_size, 100*block_mul*block_mul*number_of_inputs*number_of_dims*id_mul, 0x21000000)])
        simd0.set_up_allocators([(tt_dtype.Float16, block_size, 100*block_mul*block_mul*number_of_dims*id_mul*2, 0x31000000)])
        runtime = tt_runtime(simd0, netlist)

        lin = torch.randn((1,1,number_of_dims,512,1024))
        rin = torch.randn((1,1,number_of_dims,1024,1024))
        lin_ttens = tt_tensor(block_size=block_size, simd_cluster=simd0, torch_tensor=lin, dtype=tt_dtype.Float32)
        rin_ttens = tt_tensor(block_size=block_size, simd_cluster=simd0, torch_tensor=rin, dtype=tt_dtype.Float32)
        lin_ttens.to_device(0,lin)
        rin_ttens.to_device(0,rin)

        #genout = netlist.unary_tensor_op(tt_net_op_types.nop, lin_ttens, tt_op_dtype(tt_dtype.Float16))
        genout = ttf.matmul(lin_ttens,rin_ttens,tt_op_dtype(tt_dtype.Float16),runtime) #netlist.binary_tensor_op(tt_net_op_types.matmul, lin_ttens, rin_ttens, tt_op_dtype(tt_dtype.Float16))
        status = backend.compile_and_run_netlist(netlist.get_last_netlist_name(), {})
        assert status == BackendStatusCode.Success
        backend.wait_for_idle()

        out = genout.from_device(0)
        out = out.type(torch.float32)
        golden_out = torch.matmul(lin,rin)
        assert torch.allclose(out,golden_out,atol=0.5,rtol=0.5)
        #assert torch.allclose(out,lin,atol=0.5,rtol=0.5)

        del(lin_ttens)
        del(rin_ttens)
        del(genout)

def simd_malloc_test():
    logging.info("Testing TT SIMD Cluster Malloc Machinery!")
    simd0 = tt_simd_cluster(4,8, list(range(4*8)))

    # set up spaces to randomly sample
    num_alloc_list = [1,2,3,4,5,6,7,8,16]
    block_size_list = [32,64,128,256,512]
    num_block_list = [32]
    dram_bottom_list = [0,9,11,18]
    tensor_shape_list = [(1,1,4,4),(2,2,4,4),(1,1,16,16),(1,1,4,16),(1,1,16,4)]

    num_allocs = random.choice(num_alloc_list)
    alloc_list = []
    alloc_hash_list = []
    for alloc in range(num_allocs):
        # randomize dtype
        dtype = random.choice(list(tt_dtype))
        block_size = random.choice(block_size_list)
        num_blocks = random.choice(num_block_list)
        dram_bottom = random.choice(dram_bottom_list)
        alloc_list.append((dtype,block_size,num_blocks,dram_bottom))
        alloc_hash_list.append((block_size,dtype))

    # set up the randomly chosen allocators
    simd0.set_up_allocators(alloc_list)

    # Generate some tensors off the allocators
    for i in range(50):
        tensor_alloc_hash = random.choice(alloc_hash_list)
        tensor_dtype = tensor_alloc_hash[1]
        tensor_block_size = tensor_alloc_hash[0]
        tensor_shape = random.choice(tensor_shape_list)
        tns0 = tt_tensor(block_size=tensor_block_size, simd_cluster=simd0, shape=(1,1,4,4) , dtype=tensor_dtype)
        del tns0

    # Go through allocators and check that everything was de-allocated properly
    simd0.check_allocator_end_state()
    logging.info("Successfully allocated: ", i+1, " tensors")

def main():
    #grayskull_read_write_test()
    grayskull_matmul_test()
    duplicate_alloc_test()
    simd_malloc_test()
    grayskull_matmul_test()

if __name__ == "__main__":
    main()
