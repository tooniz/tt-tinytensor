import torch

class tt_malloc:
    def __init__(self, block_size_bytes, num_blocks, dram_bot):
        # Start address to use - applies to all dram channels
        # below this address is space reserved for program queues
        # and whatever else is not being managed by Tiny Tensor
        # allocators
        self.dram_chan_bottom = dram_bot

        # Block size associated with this allocator
        self.block_size_bytes    = block_size_bytes

        self.num_blocks = num_blocks

        # fill free list tensor with randomized but unique block indices
        # these will be used to generate physical channels and within-channel
        # addresses when reading/writing the hardware
        self.free_block_index_tensor = torch.randperm(num_blocks, dtype=torch.int64)

    def allocate_tensor(self, addr_tensor):
        addr_tensor_shape = addr_tensor.shape
        tensor_length = list(addr_tensor.flatten(start_dim=2,end_dim=-1).shape)[-1]

        assert list(self.free_block_index_tensor.shape)[0] > tensor_length, "Error: asking for more blocks than left in free list"

        # get the first 'tensor_length' elements of free list
        blocks = self.free_block_index_tensor[:tensor_length]

        # pop the first 'tensor_length' indices off the free list
        self.free_block_index_tensor = self.free_block_index_tensor[tensor_length:]

        # reshape and assign to the address tensor - allocation done!
        addr_tensor = blocks.reshape(addr_tensor.shape[2:])
        addr_tensor = addr_tensor.broadcast_to(addr_tensor_shape)
        return addr_tensor

    def deallocate_tensor(self, addr_tensor):
        # flatten the tensor to be de-allocated, and add it to the free list tensor - de-allocation done
        self.free_block_index_tensor = torch.cat((self.free_block_index_tensor, addr_tensor[0][0].flatten()))


