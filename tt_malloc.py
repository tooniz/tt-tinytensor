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

        # fill free list tensor with randomized but unique block indices
        # these will be used to generate physical channels and within-channel
        # addresses when reading/writing the hardware
        self.free_block_index_tensor = torch.randperm(num_blocks, dtype=torch.int64)

    def allocate_tensor(self, addr_tensor):
        tensor_length = list(addr_tensor.flatten().shape)[0]

        assert list(self.free_block_index_tensor.shape)[0] > tensor_length, "Error: asking for more blocks than left in free list"

        # get the first 'tensor_length' elements of free list
        blocks = self.free_block_index_tensor[:tensor_length]

        # pop the first 'tensor_length' indices off the free list
        self.free_block_index_tensor = self.free_block_index_tensor[tensor_length:]

        # reshape and assign to the address tensor - allocation done!
        addr_tensor = blocks.reshape(addr_tensor.shape)
        return addr_tensor

    def deallocate_tensor(self, addr_tensor):
        # flatten the tensor to be de-allocated, and add it to the free list tensor - de-allocation done
        self.free_block_index_tensor = torch.cat((self.free_block_index_tensor, addr_tensor.flatten()))


####
# Below here is tt_allocator test code

def main():
    print("Testing TT allocator!")
    test_allocs_followed_by_deallocs(False)
    test_allocs_followed_by_deallocs(True)

    # randomize input parameters
    # random sequence of tensor allocates and de-allocates
    # checks:
    #  - check that at the end, the free list is the right size
    #  - check that there are no duplicates in free list
    #  - check that allocator barfs if free list is over-run

def test_allocs_followed_by_deallocs(interleave):
    # random length of free list
    num_blocks = torch.randint(low=24,high=1000,size=(1,1)).item()

    # make an allocator object
    ttm = tt_malloc(32, num_blocks, 0)

    original_free_list = ttm.free_block_index_tensor
    assert(num_blocks == list(ttm.free_block_index_tensor.shape)[0])

    # Generate a bunch of small random tensors, allocate them all
    # then de-allocate them all
    alloc_blocks = 0
    alloc_tensors = 0
    tensor_list = []
    while(alloc_blocks < (num_blocks-24)):
        tpl = torch.randperm(4) + 1
        tpl = tuple(tpl.tolist())
        tns = torch.zeros(tpl, dtype=torch.int64)
        tns = ttm.allocate_tensor(tns)
        if(interleave == True):
            ttm.deallocate_tensor(tns)
        else:
            tensor_list.append(tns)
        alloc_blocks = alloc_blocks + list(tns.flatten().shape)[0]
        alloc_tensors = alloc_tensors + 1

    # de-allocate all tensors
    if(interleave == False):
        for tensor in tensor_list:
            ttm.deallocate_tensor(tensor)

    # check the free list is back to being fully free
    assert list(ttm.free_block_index_tensor.shape)[0] == num_blocks, "Error: deallocator did not put back all blocks"

    # check the blocks in the free list are unique
    unique = torch.unique(ttm.free_block_index_tensor)
    assert unique.shape == ttm.free_block_index_tensor.shape, "Error: the free list got poluted, with duplicate blocks"

    print("Passed: allocs_followed_by_deallocs, having allocated and deallocated ", alloc_tensors, " tensors")

if __name__ == "__main__":
    main()