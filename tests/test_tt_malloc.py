import torch
from tt_malloc import tt_malloc

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