# Allocator functionality:
Figure out how to hash from dram base addresses to 6 dram channels. Current implementation has only powers of two.
Consider whether the base address adding and multiply by block size should be done in tensor.get_list() as opposed to tt_malloc

# Tensor functionality

# Netlist API functionality

# SIMD Testing
Add further checking to SIMD allocator and malloc test
- check the right amount of blocks was allocated
- check that chip dimensions are truly broadcast (multiple copies of same tensor allocations for each chip)

# Low level functionality that is needed
- Light weight queue command that will leave everything the same but change dram queue addresses to read/write
- Ability to loop program on local devices without hacks
- Ability to string epoch outputs->inputs via sram



