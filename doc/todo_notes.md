#####
##### List of hacks
#####
- Sort out needing to allocate twice the calculated buffer size

# 
# High level philosophy
How to capture pipeline behaviors clean

# Allocator functionality:
- Hashing to six / 8 channels and hashed address generation for multi-channel
- Consider whether the base address adding and multiply by block size should be done in tensor.get_list() as opposed to tt_malloc

# Trace run to figure out allocators and allocation size

# Tensor functionality
- Tony will need to make such that I don't need to initialize every RAM, or else we need to support init of 8b, 4, and 2b data types.
- Hashing to six / 8 channels and hashed address generation for multi-channel

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
- Buda runtime to return stats about last run command, into a data structure that developer can parse
- Netlist to enable targetting devices that are smaller than a chip - ie nVidia MiG
- Matmul and max reduce that can keep accumulating across 'light epochs', to reduce across slice dimensions
- Ensure using RAMs is as performant for reads/writes/reblocking as with queues
- Ability to loop program on local devices without hacks
- Ability to string epoch outputs->inputs via sram


