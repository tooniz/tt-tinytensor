import torch
from tt_malloc import tt_malloc
from tt_simd_cluster import tt_simd_cluster

class tt_tensor(): 
    def __init__(self, block_size: int, allocator: tt_malloc, simd_cluster: tt_simd_cluster, torch_tensor: torch.Tensor = None, shape: tuple = None, data_format='Bfp8_b'):
        # save local references for block size, the dram allocator and chip grid
        self.allocator = allocator
        self.simd_cluster = simd_cluster
        self.block_size = block_size
        self.data_format = data_format

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


