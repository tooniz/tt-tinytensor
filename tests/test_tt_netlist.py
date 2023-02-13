def main():
    simd0  = tt_simd_cluster(2,2,(0,1,2,3))
    num_alloc_blocks = 100000
    simd0.set_up_allocators([(tt_dtype.Bfp8_b,64,num_alloc_blocks,0)])
    simd0.set_up_allocators([(tt_dtype.Bfp8_b,128,num_alloc_blocks,0)])

    netlist0 = tt_netlist()

    block_size = 128
    dim_list0 = (1,2,3,8,8)
    dim_list1 = (1,2,3,4,4)
    dim_list2 = (1,2,3,2,2)

    bla0 = tt_tensor(block_size=block_size, simd_cluster=simd0, shape=tuple(dim_list0))
    bla1 = tt_tensor(block_size=block_size, simd_cluster=simd0, shape=tuple(dim_list1))
    bla2 = tt_tensor(block_size=block_size, simd_cluster=simd0, shape=tuple(dim_list2))

    out_tensor = netlist0.binary_tensor_op(tt_net_op_types.matmul, bla0, bla1, tt_op_dtype(tt_dtype.Bfp8_b))
    netlist0.dump_netlist()

    del(bla0)
    del(bla1)
    del(bla2)
    del(out_tensor)

if __name__ == "__main__":
    print("Testing TT netlist api")

    main()

    #main()
