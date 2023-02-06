import torch
import random
import logging
from tt_simd_cluster import tt_simd_cluster
from tt_simd_cluster import tt_dtype
from tt_tensor import tt_tensor

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
    duplicate_alloc_test()
    simd_malloc_test()

if __name__ == "__main__":
    main()
