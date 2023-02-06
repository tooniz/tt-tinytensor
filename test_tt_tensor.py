import torch
from tt_tensor import tt_tensor
from tt_simd_cluster import tt_simd_cluster
from tt_simd_cluster import tt_dtype
from tt_malloc import tt_malloc

####
# Below here is tt_tensor test code
def main():
    alloc_dealloc_test()

def alloc_dealloc_test():
    print("Testing TT tensor!")

    simd0  = tt_simd_cluster(2,2,(0,1,2,3))
    num_alloc_blocks = 100000
    alloc0 = tt_malloc(32, num_alloc_blocks, 0)

    loop_cnt = 0
    while(loop_cnt < 100):
        loop_cnt = loop_cnt + 1

        # shape and block size constraints
        # first pick block size
        possible_block_size = (32,64,128,256,512,1024)
        block_size = possible_block_size[torch.randint(low=0,high=6,size=(1,1)).item()]
        simd0.set_up_allocators([(tt_dtype.Bfp8_b,block_size,100000,0)])

        # make R,C shape a multiple of the block size
        r_mul = torch.randint(low=1,high=32,size=(1,1)).item()
        c_mul = torch.randint(low=1,high=32,size=(1,1)).item()

        # make total dim count random
        dims = torch.randint(low=4,high=8,size=(1,1)).item()

        # make other dim values relatively small and random
        dim_list = []
        for i in range(dims):
            dim_list.append(torch.randint(low=1,high=4,size=(1,1)).item())
        dim_list[-1] = int(c_mul * block_size)
        dim_list[-2] = int(r_mul * block_size)

        # randomly pick pytorch tensor as input and shape tuple as input
        if(torch.randint(low=0,high=2,size=(1,1)).item()):
            dim_list[-1] = int(dim_list[-1]/block_size)
            dim_list[-2] = int(dim_list[-2]/block_size)
            bla = tt_tensor(block_size=block_size, simd_cluster=simd0, shape=tuple(dim_list))
            assert bla.addr_shape == tuple(dim_list), "tt_tensor init from shape tuple, address shape not equal to expected"
        else:
            torch_tens = torch.randn(tuple(dim_list))
            bla = tt_tensor(block_size=block_size, simd_cluster=simd0, torch_tensor=torch_tens)
            dim_list[-1] = int(dim_list[-1]/block_size)
            dim_list[-2] = int(dim_list[-2]/block_size)
            assert bla.addr_shape == tuple(dim_list), "tt_tensor init from torch tensor, address shape not equal to expected"

        # delete and de-allocate tt_tensor
        del(bla)

        # check that the allocator is back to being empty after de-allocating
        print("Allocator state:", tuple(dim_list), list(alloc0.free_block_index_tensor.shape)[0], num_alloc_blocks)
        assert list(alloc0.free_block_index_tensor.shape)[0] == num_alloc_blocks, "Error: deallocator did not put back all blocks: "

        # check the blocks in the free list are unique
        unique = torch.unique(alloc0.free_block_index_tensor)
        assert unique.shape == alloc0.free_block_index_tensor.shape, "Error: the free list got poluted, with duplicate blocks"


if __name__ == "__main__":
    main()