import torch
from tt_malloc import tt_malloc
from tt_simd_cluster import tt_simd_cluster
from tt_dtype import tt_dtype
import logging

logging.basicConfig(level=logging.DEBUG)
class tt_tensor(): 
    id = 0
    def __init__(self, block_size: int, simd_cluster: tt_simd_cluster, torch_tensor: torch.Tensor = None, shape: tuple = None, dtype=tt_dtype.Bfp8_b):
        self.id = tt_tensor.id
        tt_tensor.id = tt_tensor.id + 1
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

        # Hack to work around the need to initialize each RAM
        list_shape = list(self.shape)
        list_shape[-1] = int(list_shape[-1] * block_size)
        list_shape[-2] = int(list_shape[-2] * block_size)

        if(dtype == tt_dtype.Float32):
            self.torch_dtype = torch.float32
        elif(dtype == tt_dtype.Float16):
            self.torch_dtype = torch.float16
        elif(dtype == tt_dtype.Float16_b):
            self.torch_dtype = torch.bfloat16
        else:
            self.torch_dtype = None
        zeros = torch.full(list_shape,33.0,dtype=self.torch_dtype)
        self.to_device(0,zeros) 

    def __del__(self):
        # once tt_tensor goes out of scope
        # de-allocate space you had reserved via the attached tt allocator
        log_string = "DEALLOCATING TT TENSOR with ID: " + str(self.id)
        logging.debug(log_string)
        self.simd_cluster.deallocate_tensor(self)

    def get_dram_list(self, tensor_slice):
        flat_addr_tensor = self.address_tensor.flatten(start_dim=2,end_dim=-3)
        # bit_mask = torch.full(flat_addr_tensor[0,0,tensor_slice].shape,7,dtype=torch.int64)
        # channel = torch.bitwise_and(flat_addr_tensor[0,0,tensor_slice], bit_mask)
        # shift = torch.full((1,),3,dtype = torch.int64)
        # addr = torch.bitwise_right_shift(flat_addr_tensor[0,0,tensor_slice], shift)
        # list_a = channel.flatten().tolist()
        # list_b = addr.flatten().tolist()
        list_b = flat_addr_tensor[0,0,tensor_slice].flatten().tolist()
        list_a = [1] * len(list_b)
        return list(map(list, zip(list_a, list_b)))

    def get_dram_addr_tensor_slice(self, slice: int):
        # just return the indices for initial test
        addr_tensor_flat = self.address_tensor.flatten(start_dim=2,end_dim=-3)
        return addr_tensor_flat[0][0][slice]

    def get_dram_chan_tensor_slice(self, slice: int):
        # put everything in channel one for initial test
        chan_tens = torch.ones((self.address_tensor.shape[-2], self.address_tensor.shape[-1]),dtype=torch.int32)
        return chan_tens

    def to_device(self, chip_id: int, torch_in: torch.Tensor):
        assert self.torch_dtype is not None

        # convert input tensor to expected input data type
        torch_in_dt = torch_in.type(self.torch_dtype)

        # Generate flat view of tensor dims except for chip dims and 2D slices
        torch_in_flat = torch_in_dt.flatten(start_dim=2,end_dim=-3)
        iterations = torch_in_flat.shape[2]

        # write out all slices
        for slice in range(iterations):
            self.simd_cluster.write_tensor_slice_to_dram(chip_id=chip_id, data=self.block_tensor_slice(torch_in_flat[0][0][slice], block_dim=self.block_size), chan=self.get_dram_chan_tensor_slice(slice), address=self.get_dram_addr_tensor_slice(slice))

    def from_device(self, chip_id):
        assert self.torch_dtype is not None

        # Generate flat view of tensor dims except for chip dims and 2D slices
        addr_tensor_flat = self.address_tensor.flatten(start_dim=2,end_dim=-3)
        iterations = addr_tensor_flat.shape[2]

        # create read target tensor
        tensor_shape = list(self.address_tensor.shape)
        tensor_shape[-1] = int(tensor_shape[-1] *  self.block_size)
        tensor_shape[-2] = int(tensor_shape[-2] *  self.block_size)
        read_tensor = torch.empty(tensor_shape)
        read_tensor.type(self.torch_dtype)

        # flat view of read target tensor
        read_tensor_flat = read_tensor.flatten(start_dim=2,end_dim=-3)
        read_tensor_flat = read_tensor_flat.type(self.torch_dtype)
        # read back all slices
        for slice in range(iterations):
            read_tensor_flat[0][0][slice] = self.unblock_tensor_slice(self.simd_cluster.read_tensor_slice_from_dram(chip_id, read_tensor_flat[0][0][slice], self.get_dram_chan_tensor_slice(slice), self.get_dram_addr_tensor_slice(slice),torch_dtype=self.torch_dtype),block_dim=self.block_size)
        return read_tensor_flat

    def block_tensor_slice(self, tensor, block_dim = 128, ublock_dim = 64, tile_dim = 32, face_dim = 16):
        blocks_r = int(tensor.shape[-2] / block_dim)
        blocks_c = int(tensor.shape[-1] / block_dim)
        ublocks_r = int(block_dim / ublock_dim)
        ublocks_c = int(block_dim / ublock_dim)
        tiles_r = int(ublock_dim / tile_dim)
        tiles_c = int(ublock_dim / tile_dim)
        faces_r = int(tile_dim / face_dim)
        faces_c = int(tile_dim / face_dim)
        blocked_tensor = tensor.reshape(blocks_r,ublocks_r,tiles_r,faces_r,face_dim,blocks_c,ublocks_c,tiles_c,faces_c,face_dim)
        permuted = blocked_tensor.permute(-10,-5,-9,-4,-8,-3,-7,-2,-6,-1)
        flattened = permuted.flatten(start_dim=-10,end_dim=-1)
        back_2d = flattened.reshape(tensor.shape[-2], tensor.shape[-1])
        return back_2d

    def unblock_tensor_slice(self, tensor, block_dim = 128, ublock_dim = 64, tile_dim = 32, face_dim = 16):
        blocks_r = int(tensor.shape[-2] / block_dim)
        blocks_c = int(tensor.shape[-1] / block_dim)
        ublocks_r = int(block_dim / ublock_dim)
        ublocks_c = int(block_dim / ublock_dim)
        tiles_r = int(ublock_dim / tile_dim)
        tiles_c = int(ublock_dim / tile_dim)
        faces_r = int(tile_dim / face_dim)
        faces_c = int(tile_dim / face_dim)
        blocked_tensor = tensor.reshape(blocks_r,blocks_c,ublocks_r,ublocks_c,tiles_r,tiles_c,faces_r,faces_c,face_dim,face_dim)
        permuted = blocked_tensor.permute(-10,-8,-6,-4,-2,-9,-7,-5,-3,-1)
        flattened = permuted.flatten(start_dim=-10,end_dim=-1)
        back_2d = flattened.reshape(tensor.shape[-2], tensor.shape[-1])
        return back_2d

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


