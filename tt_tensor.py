import torch
from tt_malloc import tt_malloc
from tt_simd_cluster import tt_simd_cluster
from tt_simd_cluster import tt_dtype

class tt_tensor(): 
    def __init__(self, block_size: int, simd_cluster: tt_simd_cluster, torch_tensor: torch.Tensor = None, shape: tuple = None, dtype=tt_dtype.Bfp8_b):
        # save local references for block size, the dram allocator and chip grid
        self.simd_cluster = simd_cluster
        self.block_size = block_size
        self.virtual_block_size = block_size
        self.dtype = dtype

        # Tiny Tensors can be initialized via a torch tensor
        if(torch_tensor != None):
            # account for blocking and make dimensions -1, -2 smaller by factor of block size
            self.shape = tuple(torch_tensor.shape[:-2] + (int(torch_tensor.shape[-2]/block_size),int(torch_tensor.shape[-1]/block_size)))
        else:
            # or via a specifically supplied shape
            if(shape != None):
                self.shape = shape
            else:
                assert False, "Need to either pass a torch tensor with a reference shape, or pass an explicit shape to tt_tensor init"

        # initialize empty tensor with the right shape
        self.address_tensor = torch.empty(self.shape)

        # Call the allocator to get the new tensor filled with allocated DRAM addresses
        self.address_tensor = simd_cluster.allocate_tensor(self)

        # Initialize 'transpose lowest two dimensions' flag to False
        self.transpose_r_c = False

    def __del__(self):
        # once tt_tensor goes out of scope
        # de-allocate space you had reserved via the attached tt allocator
        self.simd_cluster.deallocate_tensor(self)

    def get_dram_list(self, tensor_slice):
        flat_addr_tensor = self.address_tensor.flatten(start_dim=2,end_dim=-3)
        bit_mask = torch.full(flat_addr_tensor[0,0,tensor_slice].shape,7,dtype=torch.int64)
        channel = torch.bitwise_and(flat_addr_tensor[0,0,tensor_slice], bit_mask)
        shift = torch.full((1,),3,dtype = torch.int64)
        addr = torch.bitwise_right_shift(flat_addr_tensor[0,0,tensor_slice], shift)
        list_a = channel.flatten().tolist()
        list_b = addr.flatten().tolist()
        return list(map(list, zip(list_a, list_b)))

    def get_dram_chan_tensor_slice(self, slice: int):
        # just return the indices for initial test
        addr_tensor_flat = self.address_tensor.flatten(start_dim=2,end_dim=-3)
        return addr_tensor_flat[0][0][slice]

    def get_dram_addr_tensor_slice(self, slice: int):
        # put everything in channel one for initial test
        chan_tens = torch.ones((address_tensor[-2], address_tensor[-1]))

    def to_device(self, chip_id: int, torch_in: torch.Tensor):
        # Check that incoming tensor dimensions match the tt_tensors

        # Generate flat view of tensor dims except for chip dims and 2D slices
        torch_in_flat = torch_in.flatten(start_dim=2,end_dim=-3)
        iterations = torch_in_flat.shape[2]

        # write out all slices
        for slice in range(iterations):
            tt_simd_cluster.write_tensor_slice_to_dram(chip_id, torch_in_flat[0][0][slice], self.get_dram_chan_tensor_slice(slice), self.get_dram_addr_tensor_slice(slice))

    def from_device(self):
        # Generate flat view of tensor dims except for chip dims and 2D slices
        addr_tensor_flat = self.address_tensor.flatten(start_dim=2,end_dim=-3)
        iterations = addr_tensor_flat.shape[2]

        # create read target tensor
        tensor_shape = list(self.addr_tensor.shape)
        tensor_shape[-1] = int(tensor_shape[-1] *  self.block_size)
        tensor_shape[-2] = int(tensor_shape[-2] *  self.block_size)
        read_tensor = torch.empty(tensor_shape)

        # flat view of read target tensor
        read_tensor_flat = read_tensor.flatten(start_dim=2,end_dim=-3)
        
        # read back all slices
        for slice in range(iterations):
            read_tensor_flat[0][0][slice] = tt_simd_cluster.read_tensor_slice_from_dram(chip_id, read_tensor_flat[0][0][slice], self.get_dram_chan_tensor_slice(slice), self.get_dram_addr_tensor_slice(slice))

        return read_tensor_flat

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


