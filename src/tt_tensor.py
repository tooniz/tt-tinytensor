import torch
import math
from tt_malloc import tt_malloc
from tt_simd_cluster import tt_simd_cluster, tt_dram_chan_picker
from eager_backend import BackendStatusCode
from tt_dtype import tt_dtype
from tt_dtype import tt_op_dtype
import logging

# logging.basicConfig(level=logging.DEBUG)
class tt_tensor(): 
    id = 0
    def __init__(self, block_size: int, simd_cluster: tt_simd_cluster, torch_tensor: torch.Tensor = None, shape: tuple = None, addr_tensor = None, dtype=tt_dtype.Float16, parent_tensor=None):
        self.id = tt_tensor.id
        tt_tensor.id = tt_tensor.id + 1
        # save local references for block size, the dram allocator and chip grid
        self.simd_cluster = simd_cluster
        assert self.simd_cluster.r != 0 and self.simd_cluster.c != 0, "Simd cluster cannot be empty"
        self.block_size = block_size
        self.dtype = dtype
        self.parent = parent_tensor
        # Initialize virtual block size and set 'transpose lowest two dimensions' flag to False
        self.virtual_block_size = block_size
        self.transpose_r_c = False
        # handle setting of torch.dtype (for to_device() and from_device())
        if(dtype == tt_dtype.Float32):
            self.torch_dtype = torch.float32
        elif(dtype == tt_dtype.Float16):
            self.torch_dtype = torch.float16
        elif(dtype == tt_dtype.Float16_b):
            self.torch_dtype = torch.bfloat16
        else:
            self.torch_dtype = None

        # Tiny Tensors can be initialized via a torch tensor
        if(torch_tensor != None):
            # account for blocking and make dimensions -1, -2 smaller by factor of block size
            self.shape = tuple(torch_tensor.shape[:-2] + (torch_tensor.shape[-2] // block_size, torch_tensor.shape[-1] // block_size))
            assert self.shape[-2] * block_size == torch_tensor.size(-2) and self.shape[-1] * block_size == torch_tensor.size(-1)
        else:
            # or via a specifically supplied shape
            if(shape != None):
                self.shape = shape
            else:
                assert False, "Need to either pass a torch tensor with a reference shape, or pass an explicit shape to tt_tensor init"

        # initialize empty tensor with the right shape
        self.address_tensor = torch.empty(self.shape)
        self.channel_tensor = torch.ones(self.shape, dtype=torch.int32)


        if(addr_tensor != None):
            self.address_tensor = addr_tensor
            self.channel_tensor = torch.ones(addr_tensor.shape, dtype=torch.int32)

        # allocate memory if you are not a 'view' of an existing tensor
        # and if the back end API 
        if(parent_tensor == None and addr_tensor == None):
            # Call the allocator to get the new tensor filled with allocated DRAM addresses
            self.address_tensor = simd_cluster.allocate_tensor(self)

            chan_slice = torch.empty((self.shape[-2], self.shape[-1]), dtype=torch.int32)
            for r in range(chan_slice.shape[-2]):
                for c in range(chan_slice.shape[-1]):
                    block_id = r * chan_slice.shape[-1] + c
                    chan_slice[r][c] = self.get_dram_chan(block_id)

            # broadcast channel assignment to all slices
            self.channel_tensor = self.channel_tensor * chan_slice

            # if a device back end is set up, initialize the allocated memory on device
            if(self.simd_cluster.be_api != None):
                # Hack to work around the need to initialize each RAM
                # list_shape = list(self.shape)
                # list_shape[-1] = int(list_shape[-1] * block_size)
                # list_shape[-2] = int(list_shape[-2] * block_size)
                # zeros = torch.full(list_shape,33.0,dtype=self.torch_dtype)
                self.init_on_devices(iterations=math.prod(self.shape[:-2]))


    def __del__(self):
        # once tt_tensor goes out of scope
        # de-allocate space you had reserved via the attached tt allocator
        # but only if the tensor is not a 'view' (ie if it's a parent,
        # that has allocated and owns the underlying storage tensor)
        if(self.parent == None):
            log_string = "DEALLOCATING TT TENSOR with ID: " + str(self.id)
            logging.debug(log_string)
            self.simd_cluster.deallocate_tensor(self)

    def get_dram_chan(self, block_id):
        if (self.simd_cluster.chan_picker == tt_dram_chan_picker.distributed):
            uniq_hash = hash(f"{self.id}_{block_id}")
            return uniq_hash % self.simd_cluster.num_dram_channels()
        elif (self.simd_cluster.chan_picker == tt_dram_chan_picker.roundrobin_block):
            return block_id % self.simd_cluster.num_dram_channels()
        elif (self.simd_cluster.chan_picker == tt_dram_chan_picker.roundrobin_tensor):
            return self.id % self.simd_cluster.num_dram_channels()
        elif (self.simd_cluster.chan_picker == tt_dram_chan_picker.constant):
            return 1
        return -1

    def get_dram_list(self, tensor_slice):
        if(self.transpose_r_c):
            # transpose the lower dims into untransposed form
            # this is since the netlist/backend will do the actual transpose via a different mechanism
            flat_addr_tensor = self.address_tensor.swapaxes(-1,-2).flatten(start_dim=0,end_dim=-3)
            flat_chan_tensor = self.channel_tensor.swapaxes(-1,-2).flatten(start_dim=0,end_dim=-3)
        else:
            flat_addr_tensor = self.address_tensor.flatten(start_dim=0,end_dim=-3)
            flat_chan_tensor = self.channel_tensor.flatten(start_dim=0,end_dim=-3)

        # bit_mask = torch.full(flat_addr_tensor[0,0,tensor_slice].shape,7,dtype=torch.int64)
        # channel = torch.bitwise_and(flat_addr_tensor[0,0,tensor_slice], bit_mask)
        # shift = torch.full((1,),3,dtype = torch.int64)
        # addr = torch.bitwise_right_shift(flat_addr_tensor[0,0,tensor_slice], shift)
        # list_a = channel.flatten().tolist()
        # list_b = addr.flatten().tolist()
        list_b = flat_addr_tensor[tensor_slice].flatten().tolist()
        list_a = flat_chan_tensor[tensor_slice].flatten().tolist()
        assert len(list_a) == len(list_b)
        return list(map(list, zip(list_a, list_b)))

    def get_dram_addr_tensor_slice(self, slice: int):
        # just return the indices for initial test
        addr_tensor_flat = self.address_tensor.flatten(start_dim=0,end_dim=-3)
        return addr_tensor_flat[slice].contiguous()

    def get_dram_chan_tensor_slice(self, slice: int):
        # put everything in channel one for initial test
        chan_tensor_flat = self.channel_tensor.flatten(start_dim=0,end_dim=-3)
        return chan_tensor_flat[slice].contiguous()

    # Obsolete to_devices()
    # sends to all devices that the tensor maps onto
    # def to_devices(self, torch_in: torch.Tensor):
    #     assert self.torch_dtype is not None

    #     # convert input tensor to expected input data type
    #     torch_in_dt = torch_in.type(self.torch_dtype)

    #     # Generate flat view of tensor dims except for chip dims and 2D slices
    #     torch_in_flat = torch_in_dt.flatten(start_dim=0,end_dim=-3)
    #     iterations = torch_in_flat.shape[0]
    #     dim_chip_r = self.simd_cluster.r
    #     dim_chip_c = self.simd_cluster.c

    #     # write out all slices
    #     for slice in range(iterations):
    #         for chip_r in range(dim_chip_r):
    #             for chip_c in range(dim_chip_c):
    #                 # TODO: convert logical ids to physical ids
    #                 chip_id = self.simd_cluster.get_chip_id(chip_r, chip_c)
    #                 self.simd_cluster.write_tensor_slice_to_dram(chip_id=chip_id, data=self.block_tensor_slice(torch_in_flat[slice], block_dim=self.block_size), chan=self.get_dram_chan_tensor_slice(slice), address=self.get_dram_addr_tensor_slice(slice))

    def init_on_devices(self, iterations: int):
        assert self.simd_cluster.r != 0 and self.simd_cluster.c != 0, "Writing to empty cluster"
        dim_chip_r = self.simd_cluster.r
        dim_chip_c = self.simd_cluster.c
        # init all slices
        for slice in range(iterations):
            for chip_r in range(dim_chip_r):
                for chip_c in range(dim_chip_c):
                    chip_id = self.simd_cluster.get_chip_id(chip_r, chip_c)
                    self.simd_cluster.init_tensor_slice_in_dram(chip_id=chip_id, chan=self.get_dram_chan_tensor_slice(slice), address=self.get_dram_addr_tensor_slice(slice))

    def to_device(self, chip_id: int, torch_in: torch.Tensor):
        assert self.torch_dtype is not None

        # convert input tensor to expected input data type
        torch_in_dt = torch_in.type(self.torch_dtype)

        # Generate flat view of tensor dims except for chip dims and 2D slices
        torch_in_flat = torch_in_dt.flatten(start_dim=0,end_dim=-3)
        iterations = torch_in_flat.shape[0]

        assert self.simd_cluster.r != 0 and self.simd_cluster.c != 0, "Writing to empty cluster"

        # write out all slices
        for slice in range(iterations):
            self.simd_cluster.write_tensor_slice_to_dram(chip_id=chip_id, data=self.block_tensor_slice(torch_in_flat[slice], block_dim=self.block_size), chan=self.get_dram_chan_tensor_slice(slice), address=self.get_dram_addr_tensor_slice(slice))

        # assuming the tensor ram is initialized in other chips
        # broadcast the tensor to all chips via netlist
        runtime = self.simd_cluster.runtime
        if(self.simd_cluster.r != 1 or self.simd_cluster.c != 1):
            self.simd_cluster.netlist.unary_tensor_bcast_op(self, self, chip_id)
            status = runtime.backend.compile_and_run_netlist(self.simd_cluster.netlist.get_last_netlist_name(), {})
            assert status == BackendStatusCode.Success
            runtime.backend.wait_for_idle()

        return self

    # Obsolete from_devices()
    #
    # # reads from all devices tensor maps onto
    # def from_devices(self):
    #     assert self.torch_dtype is not None

    #     # Generate flat view of tensor dims except for chip dims and 2D slices
    #     addr_tensor_flat = self.address_tensor.flatten(start_dim=0,end_dim=-3).contiguous()
    #     iterations = addr_tensor_flat.shape[0]
    #     dim_chip_r = self.simd_cluster.r
    #     dim_chip_c = self.simd_cluster.c

    #     # create read target tensor
    #     tensor_shape = list(self.address_tensor.shape)
    #     tensor_shape[-1] = int(tensor_shape[-1] *  self.block_size)
    #     tensor_shape[-2] = int(tensor_shape[-2] *  self.block_size)
    #     read_tensor = torch.empty(tensor_shape).contiguous()
    #     read_tensor.type(self.torch_dtype)

    #     # flat view of read target tensor
    #     read_tensor_flat = read_tensor.flatten(start_dim=2,end_dim=-3)
    #     read_tensor_flat = read_tensor_flat.type(self.torch_dtype)
    #     # read back all slices
    #     for slice in range(iterations):
    #         for chip_r in range(dim_chip_r):
    #             for chip_c in range(dim_chip_c):
    #                 # TODO: convert logical ids to physical ids
    #                 chip_id = chip_r * dim_chip_r + chip_c
    #                 read_tensor_flat[chip_r][chip_c][slice] = self.unblock_tensor_slice(self.simd_cluster.read_tensor_slice_from_dram(chip_id, read_tensor_flat[0][0][slice], self.get_dram_chan_tensor_slice(slice), self.get_dram_addr_tensor_slice(slice),torch_dtype=self.torch_dtype),block_dim=self.block_size)
    #     return read_tensor_flat

    def from_device(self, chip_id):
        assert self.torch_dtype is not None

        # Generate flat view of tensor dims except for chip dims and 2D slices
        addr_tensor_flat = self.address_tensor.flatten(start_dim=0,end_dim=-3).contiguous()
        iterations = addr_tensor_flat.shape[0]

        # create read target tensor
        tensor_shape = list(self.address_tensor.shape)
        tensor_shape[-1] = int(tensor_shape[-1] *  self.block_size)
        tensor_shape[-2] = int(tensor_shape[-2] *  self.block_size)
        read_tensor = torch.empty(tensor_shape).contiguous()
        read_tensor.type(self.torch_dtype)

        # flat view of read target tensor
        read_tensor_flat = read_tensor.flatten(start_dim=0,end_dim=-3)
        read_tensor_flat = read_tensor_flat.type(self.torch_dtype)
        # read back all slices
        for slice in range(iterations):
            read_tensor_flat[slice] = self.unblock_tensor_slice(self.simd_cluster.read_tensor_slice_from_dram(chip_id, read_tensor_flat[slice], self.get_dram_chan_tensor_slice(slice), self.get_dram_addr_tensor_slice(slice),torch_dtype=self.torch_dtype),block_dim=self.block_size)
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

    def broadcast_to(self, bcast_spec):
        new_tensor = tt_tensor(simd_cluster=self.simd_cluster, block_size=self.block_size, shape=self.shape, dtype=self.dtype, parent_tensor=self)
        new_tensor.transpose_r_c = self.transpose_r_c
        new_tensor.address_tensor = self.address_tensor.broadcast_to(bcast_spec)
        new_tensor.channel_tensor = self.channel_tensor.broadcast_to(bcast_spec)
        new_tensor.address_tensor = new_tensor.address_tensor.type(torch.int32)
        new_tensor.channel_tensor = new_tensor.channel_tensor.type(torch.int32)
        assert new_tensor.address_tensor.dtype == torch.int32
        assert new_tensor.channel_tensor.dtype == torch.int32
        new_tensor.shape = tuple(new_tensor.address_tensor.shape)
        return new_tensor

    def reshape(self, reshape_spec):
        new_tensor = tt_tensor(simd_cluster=self.simd_cluster, block_size=self.block_size, shape=self.shape, dtype=self.dtype, parent_tensor=self)
        new_tensor.transpose_r_c = self.transpose_r_c
        new_tensor.address_tensor = self.address_tensor.reshape(reshape_spec)
        new_tensor.channel_tensor = self.channel_tensor.reshape(reshape_spec)
        new_tensor.address_tensor = new_tensor.address_tensor.type(torch.int32)
        new_tensor.channel_tensor = new_tensor.channel_tensor.type(torch.int32)
        assert new_tensor.address_tensor.dtype == torch.int32
        assert new_tensor.channel_tensor.dtype == torch.int32
        new_tensor.shape = tuple(new_tensor.address_tensor.shape)
        return new_tensor

    def unsqueeze(self, dim):
        new_tensor = tt_tensor(simd_cluster=self.simd_cluster, block_size=self.block_size, shape=self.shape, dtype=self.dtype, parent_tensor=self)
        new_tensor.transpose_r_c = self.transpose_r_c
        new_tensor.address_tensor = self.address_tensor.unsqueeze(dim)
        new_tensor.channel_tensor = self.channel_tensor.unsqueeze(dim)
        new_tensor.address_tensor = new_tensor.address_tensor.type(torch.int32)
        new_tensor.channel_tensor = new_tensor.channel_tensor.type(torch.int32)
        assert new_tensor.address_tensor.dtype == torch.int32
        assert new_tensor.channel_tensor.dtype == torch.int32
        new_tensor.shape = tuple(new_tensor.address_tensor.shape)
        return new_tensor

    def expand(self, expand_spec):
        new_tensor = tt_tensor(simd_cluster=self.simd_cluster, block_size=self.block_size, shape=self.shape, dtype=self.dtype, parent_tensor=self)
        new_tensor.transpose_r_c = self.transpose_r_c
        new_tensor.address_tensor = self.address_tensor.expand(expand_spec)
        new_tensor.channel_tensor = self.channel_tensor.expand(expand_spec)
        new_tensor.address_tensor = new_tensor.address_tensor.type(torch.int32)
        new_tensor.channel_tensor = new_tensor.channel_tensor.type(torch.int32)
        assert new_tensor.address_tensor.dtype == torch.int32
        assert new_tensor.channel_tensor.dtype == torch.int32
        new_tensor.shape = tuple(new_tensor.address_tensor.shape)
        return new_tensor

    def swapaxes(self, ax0, ax1):
        new_tensor = tt_tensor(simd_cluster=self.simd_cluster, block_size=self.block_size, shape=self.shape, dtype=self.dtype, parent_tensor=self)
        new_tensor.transpose_r_c = self.transpose_r_c
        new_tensor.address_tensor = self.address_tensor.swapaxes(ax0,ax1)
        new_tensor.channel_tensor = self.channel_tensor.swapaxes(ax0,ax1)
        new_tensor.address_tensor = new_tensor.address_tensor.type(torch.int32)
        new_tensor.channel_tensor = new_tensor.channel_tensor.type(torch.int32)
        assert new_tensor.address_tensor.dtype == torch.int32
        assert new_tensor.channel_tensor.dtype == torch.int32
        new_tensor.shape = tuple(new_tensor.address_tensor.shape)
        return new_tensor

    def permute(self, permute_spec):
        new_tensor = tt_tensor(simd_cluster=self.simd_cluster, block_size=self.block_size, shape=self.shape, dtype=self.dtype, parent_tensor=self)
        new_tensor.transpose_r_c = self.transpose_r_c
        new_tensor.address_tensor = self.address_tensor.permute(permute_spec)
        new_tensor.channel_tensor = self.channel_tensor.permute(permute_spec)
        new_tensor.address_tensor = new_tensor.address_tensor.type(torch.int32)
        new_tensor.channel_tensor = new_tensor.channel_tensor.type(torch.int32)
        assert new_tensor.address_tensor.dtype == torch.int32
        assert new_tensor.channel_tensor.dtype == torch.int32
        new_tensor.shape = tuple(new_tensor.address_tensor.shape)
        return new_tensor

    def squeeze(self, dim):
        new_tensor = tt_tensor(simd_cluster=self.simd_cluster, block_size=self.block_size, shape=self.shape, dtype=self.dtype, parent_tensor=self)
        new_tensor.transpose_r_c = self.transpose_r_c
        new_tensor.address_tensor = self.address_tensor.squeeze(dim)
        new_tensor.channel_tensor = self.channel_tensor.squeeze(dim)
        new_tensor.address_tensor = new_tensor.address_tensor.type(torch.int32)
        new_tensor.channel_tensor = new_tensor.channel_tensor.type(torch.int32)
        assert new_tensor.address_tensor.dtype == torch.int32
        assert new_tensor.channel_tensor.dtype == torch.int32
        new_tensor.shape = tuple(new_tensor.address_tensor.shape)
        return new_tensor

    def view(self, view_spec):
        new_tensor = tt_tensor(simd_cluster=self.simd_cluster, block_size=self.block_size, shape=self.shape, dtype=self.dtype, parent_tensor=self)
        new_tensor.transpose_r_c = self.transpose_r_c
        new_tensor.address_tensor = self.address_tensor.view(view_spec)
        new_tensor.channel_tensor = self.channel_tensor.view(view_spec)
        new_tensor.address_tensor = new_tensor.address_tensor.type(torch.int32)
        new_tensor.channel_tensor = new_tensor.channel_tensor.type(torch.int32)
        assert new_tensor.address_tensor.dtype == torch.int32
        assert new_tensor.channel_tensor.dtype == torch.int32
        new_tensor.shape = tuple(new_tensor.address_tensor.shape)
        return new_tensor

    def transpose(self):
        new_tensor = tt_tensor(simd_cluster=self.simd_cluster, block_size=self.block_size, shape=self.shape, dtype=self.dtype, parent_tensor=self)
        new_tensor.transpose_r_c = self.transpose_r_c
        new_tensor.address_tensor = self.address_tensor.swapaxes(-2,-1)
        new_tensor.channel_tensor = self.channel_tensor.swapaxes(-2,-1)
        new_tensor.address_tensor = new_tensor.address_tensor.type(torch.int32)
        new_tensor.channel_tensor = new_tensor.channel_tensor.type(torch.int32)
        assert new_tensor.address_tensor.dtype == torch.int32
        assert new_tensor.channel_tensor.dtype == torch.int32
        new_tensor.shape = tuple(new_tensor.address_tensor.shape)
        new_tensor.transpose_r_c = not new_tensor.transpose_r_c
        return new_tensor

    def stride(self):
        return self.address_tensor.stride()


    def calculate_shard_shape_and_stride(self, input_shape, input_shape_with_chips, input_strides, original_shape, original_strides, split_dim, split_chips):
        # only accept negative number dimension representation
        # 
        assert split_dim <= 0 

        new_shape = input_shape.copy()
        new_shape_with_chips = input_shape_with_chips.copy()
        new_strides = input_strides.copy()
        unsqueezed_old_strides = [original_shape[0] * original_strides[0]] + original_strides

        new_shape[split_dim] = int(input_shape[split_dim] / split_chips)
        new_shape_with_chips[split_dim] = int(input_shape[split_dim] / split_chips)
        new_shape_with_chips.insert(0, split_chips)
        new_strides.insert(0, int((unsqueezed_old_strides[split_dim-1]) / split_chips))

        return new_shape, new_shape_with_chips, new_strides

    def shard(self, rsplit_dim, csplit_dim):
        new_shape, new_shape_with_chips, new_strides = self.calculate_shard_shape_and_stride(list(self.shape), list(self.shape), list(self.address_tensor.stride()), list(self.shape), list(self.address_tensor.stride()), csplit_dim, self.simd_cluster.c)
        new_shape, new_shape_with_chips, new_strides = self.calculate_shard_shape_and_stride(new_shape, new_shape_with_chips, new_strides, list(self.shape), list(self.address_tensor.stride()), rsplit_dim, self.simd_cluster.r)

        # Make properly reshaped address tensor
        reshaped_address_tensor = self.address_tensor.as_strided(size=new_shape_with_chips,stride=new_strides)

        # Make new tensor
        sharded_tensor = tt_tensor(simd_cluster=self.simd_cluster, block_size=self.block_size, shape=new_shape, dtype=self.dtype)

        # Issue copy commands to fill new tensor from reshaped old one
        runtime = self.simd_cluster.runtime
        for r in range(reshaped_address_tensor.shape[0]):
            for c in range(reshaped_address_tensor.shape[1]):
                chip_id = self.simd_cluster.get_chip_id(r,c)
                self.simd_cluster.netlist.unary_copy_op(tt_tensor(addr_tensor=reshaped_address_tensor[r][c], shape=new_shape, simd_cluster=self.simd_cluster, block_size=self.block_size, dtype=self.dtype, parent_tensor=self), sharded_tensor, chip_id)
                status = runtime.backend.compile_and_run_netlist(self.simd_cluster.netlist.get_last_netlist_name(), {})
                assert status == BackendStatusCode.Success
                runtime.backend.wait_for_idle()
        return sharded_tensor

    def unshard(self, rsplit_dim, csplit_dim):
        # only accept negative format dimension index
        assert rsplit_dim <= 0 or rsplit_dim == None
        assert csplit_dim <= 0 or csplit_dim == None
        assert csplit_dim != None or rsplit_dim != None

        # Figure out new tensor shape and strides
        unsharded_shape = list(self.shape)
        unsharded_shape[rsplit_dim] = int(self.shape[rsplit_dim] * self.simd_cluster.r)
        unsharded_shape[csplit_dim] = int(self.shape[csplit_dim] * self.simd_cluster.c)

        # Make new tensor
        unsharded_tensor = tt_tensor(simd_cluster=self.simd_cluster, block_size=self.block_size, shape=unsharded_shape, dtype=self.dtype)

        # Make properly reshaped address tensor for receiving shard back into
        new_shape, new_shape_with_chips, new_strides = self.calculate_shard_shape_and_stride(unsharded_shape, unsharded_shape, list(unsharded_tensor.address_tensor.stride()), unsharded_shape, list(unsharded_tensor.address_tensor.stride()), csplit_dim, self.simd_cluster.c)
        new_shape, new_shape_with_chips, new_strides = self.calculate_shard_shape_and_stride(new_shape, new_shape_with_chips, new_strides, unsharded_shape, list(unsharded_tensor.address_tensor.stride()), rsplit_dim, self.simd_cluster.r)
        reshaped_address_tensor = unsharded_tensor.address_tensor.as_strided(size=new_shape_with_chips,stride=new_strides)

        # Broadcast the shard to the right address on each chip
        runtime = self.simd_cluster.runtime
        for r in range(reshaped_address_tensor.shape[0]):
            for c in range(reshaped_address_tensor.shape[1]):
                src_chip = self.simd_cluster.get_chip_id(r,c)
                self.simd_cluster.netlist.unary_tensor_bcast_op(
                    self, # input
                    tt_tensor(addr_tensor=reshaped_address_tensor[r][c], shape=unsharded_shape, simd_cluster=self.simd_cluster, block_size=unsharded_tensor.block_size, dtype=unsharded_tensor.dtype, parent_tensor=unsharded_tensor),
                    src_chip)
                status = runtime.backend.compile_and_run_netlist(self.simd_cluster.netlist.get_last_netlist_name(), {})
                assert status == BackendStatusCode.Success
                runtime.backend.wait_for_idle()

        return unsharded_tensor


