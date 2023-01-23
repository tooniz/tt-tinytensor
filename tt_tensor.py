import torch
from tt_malloc import tt_malloc
from tt_simd_cluster import tt_simd_cluster

class tt_tensor(): 
    def __init__(self, block_size: int, allocator: tt_malloc, simd_cluster: tt_simd_cluster, torch_tensor: torch.Tensor = None, shape: tuple = None):
        # save local references for block size, the dram allocator and chip grid
        self.allocator = allocator
        self.simd_cluster = simd_cluster
        self.block_size = block_size
        # Tiny Tensors can be initialized via a torch tensor
        if(torch_tensor != None):
            # account for blocking and make dimensions -1, -2 smaller by factor of block size
            self.addr_shape = tuple(torch_tensor.shape[:-2] + (int(torch_tensor.shape[-2]/block_size),int(torch_tensor.shape[-1]/block_size)))
        else:
            # or via a specifically supplied shape
            if(shape != None):
                self.addr_shape = shape
            else:
                assert False, "Need to either pass a torch tensor with a reference shape, or pass an explicit shape to tt_tensor init"

        # initialize empty tensor with the right shape
        self.address_tensor = torch.empty(self.addr_shape)

        # Call the allocator to get the new tensor filled with allocated DRAM addresses
        self.address_tensor = allocator.allocate_tensor(self.address_tensor)

        # Initialize 'transpose lowest two dimensions' flag to False
        self.transpose_r_c = False

    def __del__(self):
        # once tt_tensor goes out of scope
        # de-allocate space you had reserved via the attached tt allocator
        self.allocator.deallocate_tensor(self.address_tensor)

    def to_device(self, torch_in: torch.Tensor):
        # go through all chips and call tilize
        pass

    def from_device(self):
        # collect tensor back from chips
        pass

    def deallocate(self):
        self.allocator.deallocate_tensor(self.address_tensor)

    def split(self):
        pass

    def view(self):
        # View changes on the Torch tensor are propagated to the Tensor of block addresses
        # allowing copy-free view changes on the sharded on-device tensors so long as blocks
        # remain atomic
        pass
    def permute(self):
        # Permute allows interchanging dimensions of the Torch tensor
        # the block address tensor will follow original torch Tensor permutes
        # allowing copy-free dimension permutes on the sharded on-device tensors so long as blocks
        # remain atomic.
        # If the bottom two axis are permuted, this is a standard 2D transpose - and will be captured
        # as a transpose flag (self.bottom_dim_transpose_flag=True)
        # Transposed axis is supported in Tenstorrent netlists
        pass


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
        # make R,C shape a multiple of the block size
        r_mul = torch.randint(low=1,high=32,size=(1,1)).item()
        c_mul = torch.randint(low=1,high=32,size=(1,1)).item()

        # make total dim count random
        dims = torch.randint(low=2,high=6,size=(1,1)).item()

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
            bla = tt_tensor(block_size=block_size, allocator=alloc0, simd_cluster=simd0, torch_tensor=None, shape=tuple(dim_list))
            assert bla.addr_shape == tuple(dim_list), "tt_tensor init from shape tuple, address shape not equal to expected"
        else:
            torch_tens = torch.randn(tuple(dim_list))
            bla = tt_tensor(block_size=block_size, allocator=alloc0, simd_cluster=simd0, torch_tensor=torch_tens, shape=None)
            dim_list[-1] = int(dim_list[-1]/block_size)
            dim_list[-2] = int(dim_list[-2]/block_size)
            assert bla.addr_shape == tuple(dim_list), "tt_tensor init from torch tensor, address shape not equal to expected"

        # delete and de-allcotae tt_tensor
        del(bla)

        # check that the allocator is back to being empty after de-allocating
        print("Allocator state:", tuple(dim_list), list(alloc0.free_block_index_tensor.shape)[0], num_alloc_blocks)
        assert list(alloc0.free_block_index_tensor.shape)[0] == num_alloc_blocks, "Error: deallocator did not put back all blocks: "

        # check the blocks in the free list are unique
        unique = torch.unique(alloc0.free_block_index_tensor)
        assert unique.shape == alloc0.free_block_index_tensor.shape, "Error: the free list got poluted, with duplicate blocks"


if __name__ == "__main__":
    main()