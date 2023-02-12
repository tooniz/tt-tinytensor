#####
##### Sort out requeting one o two slots in Tonys code and sizing of blocks in

# 
# High level philosophy
How to capture pipeline behaviors clean

# Allocator functionality:
Figure out how to hash from dram base addresses to 6 dram channels. Current implementation has only powers of two.
Consider whether the base address adding and multiply by block size should be done in tensor.get_list() as opposed to tt_malloc

# Tensor functionality
Sort out ownership of the address mapping, who and when calls deallocate, in the presence of multiple TT views of the same data

# Netlist API functionality

# SIMD Testing
Add further checking to SIMD allocator and malloc test
- check the right amount of blocks was allocated
- check that chip dimensions are truly broadcast (multiple copies of same tensor allocations for each chip)

# Netlist testing


# Low level functionality that is needed
- Graphs with multiple target devices for netlist brevity
- Light weight queue command that will leave everything the same but change dram queue addresses to read/write
- Sync primitives so that working with RAMs can be done safely
- Matmul and max reduce that can keep accumulating across 'light epochs', to reduce across slice dimensions
- Ensure using RAMs is as performant for reads/writes/reblocking as with queues
- Ability to loop program on local devices without hacks
- Ability to string epoch outputs->inputs via sram


