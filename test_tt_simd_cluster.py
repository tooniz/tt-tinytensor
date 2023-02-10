import torch
import random
import logging
from tt_simd_cluster import tt_simd_cluster
from tt_simd_cluster import tt_dtype, tt_op_dtype
from tt_tensor import tt_tensor
import IPython

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
    import eager_backend.backend_api as be_api
    from test_utils import py_desc, py_tensor
    from eager_backend import DataFormat, BackendType, BackendDevice, BackendStatusCode, IOType, IOLocation
    from eager_backend.backend_api import Backend, BackendConfig, PytorchTensorDesc

    target_arch = BackendDevice.Grayskull
    target_devices = {0}
    config = be_api.get_runtime_config(target_arch)
    backend = Backend(config, target_devices)
    be_api.initialize_child_process(target_arch, target_devices)

    simd0 = tt_simd_cluster(4,8, list(range(4*8)), be_api)
    simd0.set_up_allocators([(tt_dtype.Float32, 128, 10000, 0x20000000)])
    simd0.set_up_allocators([(tt_dtype.Float32, 64, 10000, 0x20000000)])
    simd0.set_up_allocators([(tt_dtype.Float32, 32, 10000, 0x20000000)])

    for i in range(8):
        dims = random.choice([1,3,4])
        block_size = random.choice([32,64,128])
        tensor_size = random.choice([256,128,512])
        tens = torch.randn((1,1,dims,tensor_size,tensor_size))
        tt_tens = tt_tensor(block_size=block_size, simd_cluster=simd0, torch_tensor=tens, dtype=tt_dtype.Float32)
        tt_tens.to_device(0,tens)
        ble = tt_tens.from_device(0)
        diff = torch.isclose(tens,ble)
        inv = torch.logical_not(diff)
        indices = inv.nonzero()
        del(tt_tens)
        #IPython.embed()
        assert torch.allclose(tens,ble)

    simd0.be_api.finish_child_process()
    #backend.destroy()
    print("Success")

def grayskull_matmul_test():
    import time
    import torch
    from tt_netlist import tt_netlist
    from tt_netlist import tt_net_op_types
    import eager_backend.backend_api as be_api
    from test_utils import py_desc, py_tensor
    from eager_backend import DataFormat, BackendType, BackendDevice, BackendStatusCode, IOType, IOLocation
    from eager_backend.backend_api import Backend, BackendConfig, PytorchTensorDesc

    target_arch = BackendDevice.Grayskull
    target_devices = {0}
    config = be_api.get_runtime_config(target_arch)
    backend = Backend(config, target_devices)
    be_api.initialize_child_process(target_arch, target_devices)

    simd0 = tt_simd_cluster(4,8, list(range(4*8)), be_api)
    netlist = tt_netlist()

    # set up allocators with a free list of 1 block, so as to ensure the addresses of
    # allocated blocks are equal to the dram_bottom address provided
    # this is needed since the netlist this test is running is hand coded with 
    # fixed input and output addresses
    simd0.set_up_allocators([(tt_dtype.Float32, 128, 1, 0x21000000)])
    simd0.set_up_allocators([(tt_dtype.Float16, 128, 2, 0x31000000)])

    # make a one block tensor of ones
    # having it be all ones side steps questions about tiling/ublocks/mblocks/etc
    tens = torch.randn((1,1,1,128,128))
    tt_act_tens = tt_tensor(block_size=128, simd_cluster=simd0, torch_tensor=tens, dtype=tt_dtype.Float32)
    tt_output_tens = tt_tensor(block_size=128, simd_cluster=simd0, torch_tensor=tens, dtype=tt_dtype.Float16)
    print(tt_act_tens.address_tensor)
    print(tt_output_tens.address_tensor)
    tt_act_tens.to_device(0,tens)
    tt_output_tens.to_device(0,tens)

    genout = netlist.binary_tensor_op(tt_net_op_types.matmul, tt_act_tens, tt_act_tens, tt_op_dtype(tt_dtype.Float16))

    out = tt_output_tens.from_device(0)
    assert torch.allclose(out,tens)

    status = backend.compile_and_run_netlist("loader/tests/net_basic/netlist_eager_mm_gen.yaml", {})
    assert status == BackendStatusCode.Success
    print("before Wait for idle DONE")

    backend.wait_for_idle()
    print("Wait for idle DONE")
    simd0.be_api.finish_child_process()
    backend.destroy()
    print("Success")

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
    for allocator in simd0.allocators.values():
        # check the free list is back to being fully free
        assert list(allocator.free_block_index_tensor.shape)[0] == allocator.num_blocks, "Error: deallocator did not put back all blocks"

        # check the blocks in the free list are unique
        unique = torch.unique(allocator.free_block_index_tensor)
        assert unique.shape == allocator.free_block_index_tensor.shape, "Error: the free list got poluted, with duplicate blocks"
    logging.info("Successfully allocated: ", i+1, " tensors")

def main():
    grayskull_matmul_test()
    #grayskull_read_write_test()
    #duplicate_alloc_test()
    #simd_malloc_test()

if __name__ == "__main__":
    main()
