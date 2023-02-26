import torch
import random
import logging
from tt_simd_cluster import tt_simd_cluster
from tt_simd_cluster import tt_dtype, tt_op_dtype
from tt_tensor import tt_tensor
import time
import torch
from tt_netlist import tt_netlist
from tt_netlist import tt_net_op_types
from tt_runtime import tt_runtime
import tt_functional as ttf
import eager_backend.backend_api as be_api
from test_utils import py_desc, py_tensor
from eager_backend import DataFormat, BackendType, BackendDevice, BackendStatusCode, IOType, IOLocation
from eager_backend.backend_api import Backend, BackendConfig, PytorchTensorDesc
from IPython import embed
import tt_functional as ttf
from tt_netlist import tt_net_op_types
from tt_dtype import tt_op_dtype
from tt_dtype import tt_dtype


def gen_random_inputs(mm = False):
    # Want small inputs, large inputs that fit onto a chip, big inputs that need folding along
    # all folding dimensions. Want inputs that need broadcasting to work
    #
    # Strategy:
    # - Generate a randmom sized block
    # - Generate a random sized tile as a multiplier of block dimensions. The tile is rectanuglar, not square.
    # - Generate folding multipliers
    #
    # - For matmuls, use transposed tile for left input, that along with column folding multiplier will exercise
    #   full range of shapes
    # - For non-matmuls use identical dims
    #
    # - insert random numbers of dimensions in lin and rin
    # - run unary ops on left input
    #
    # - randomize data formats, check allocation sizes
    #
    #   Run all tt_functional ops every time

    block_size = random.choice([64,128])#,256])
    tile_r = random.choice([2,8]) #,5,6,7,8])
    tile_c = random.choice([2,4,8]) #,5,6,7,8])
    tile_o_c = random.choice([2,4,8]) #,5,6,7,8])

    row_fold = random.choice([1,2])
    col_fold = random.choice([1,2])
    id_fold =  random.choice([1,2])

    mm_l_full_tile_dim_r = block_size * tile_r * row_fold
    mm_l_full_tile_dim_c = block_size * tile_c * id_fold
    mm_r_full_tile_dim_c = block_size * tile_o_c * col_fold
    linmm = torch.randn((1,1,1,1,mm_l_full_tile_dim_r,mm_l_full_tile_dim_c))
    rinmm = torch.randn((1,1,1,1,mm_l_full_tile_dim_c,mm_r_full_tile_dim_c))

    l_full_tile_dim_r = block_size * tile_r * row_fold
    l_full_tile_dim_c = block_size * tile_c * col_fold
    lin = torch.randn((1,1,1,l_full_tile_dim_r,l_full_tile_dim_c))
    rin = torch.randn((1,1,1,2,l_full_tile_dim_r,l_full_tile_dim_c))

    ##### THIS FAILED FOR SOME REASON
    if(random.choice([0,1])):
        pass

    # with open(random_inputs.txt, 'w') as out:
    #     out.write(lin.shape, rin.shape, linmm.shape, rinmm.shape, block_size, (row_fold, col_fold, id_fold, "\n")
    assert lin.shape[-1] % col_fold == 0
    assert lin.shape[-2] % row_fold == 0
    assert rin.shape[-1] % col_fold == 0
    assert rin.shape[-2] % row_fold == 0

    assert linmm.shape[-2] % row_fold == 0
    assert rinmm.shape[-1] % col_fold == 0
    assert rinmm.shape[-2] % id_fold == 0

    return lin, rin, linmm, rinmm, block_size, (row_fold, col_fold, id_fold)

def test_ops(simd0, netlist, runtime, backend, be_api):
    simd0.set_up_allocators([(tt_dtype.Float32, 64, 10000, 0x11000000)])
    simd0.set_up_allocators([(tt_dtype.Float16, 64, 10000, 0x21000000)])
    simd0.set_up_allocators([(tt_dtype.Float16, 128, 10000, 0x31000000)])
    simd0.set_up_allocators([(tt_dtype.Float32, 128, 10000, 0x36000000)])

    ##
    # Generate inputs
    ##
    lin, rin, linmm, rinmm, block_size, fold_factors = gen_random_inputs()
    lin_ttens = tt_tensor(block_size=block_size, simd_cluster=runtime.simd_cluster, torch_tensor=lin, dtype=tt_dtype.Float32)
    rin_ttens = tt_tensor(block_size=block_size, simd_cluster=runtime.simd_cluster, torch_tensor=rin, dtype=tt_dtype.Float32)
    linmm_ttens = tt_tensor(block_size=block_size, simd_cluster=runtime.simd_cluster, torch_tensor=linmm, dtype=tt_dtype.Float32)
    rinmm_ttens = tt_tensor(block_size=block_size, simd_cluster=runtime.simd_cluster, torch_tensor=rinmm, dtype=tt_dtype.Float32)
    lin_ttens.to_device(0,lin)
    rin_ttens.to_device(0,rin)
    linmm_ttens.to_device(0,linmm)
    rinmm_ttens.to_device(0,rinmm)

    ##
    # TT Functional function list
    ##
    op_dtype = tt_op_dtype(tt_dtype.Float16)
    ttf_binary_functions = [ttf.matmul, ttf.add, ttf.multiply, ttf.subtract] 
    ttf_unary_functions = [ttf.exp] #, ttf.reciprocal]
    ttf_reduction_functions = [] #[ttf.reduce]

    ##
    # Run computation and check results
    ##
    for i in range(len(ttf_binary_functions)):
        opname = ttf_binary_functions[i].__name__
        print("Op name: ", opname," ", fold_factors[0], fold_factors[1], fold_factors[2], linmm.shape, rinmm.shape)
        if(opname == "matmul"): # matmul is a special case
            print("SHAPE:",linmm.shape, rinmm.shape)
            out_ttens = ttf.matmul(linmm_ttens, rinmm_ttens, op_dtype, runtime, fold_factors)
            out = out_ttens.from_device(0)
            out = out.type(torch.float32)
            golden = ttf.matmul(linmm, rinmm)
            del(out_ttens)
        else:
            out_ttens = ttf_binary_functions[i](lin_ttens, rin_ttens, op_dtype, runtime, fold_factors)
            golden = ttf_binary_functions[i](lin, rin)
            out = out_ttens.from_device(0)
            out = out.type(torch.float32)
            del(out_ttens)
        max_diff = torch.max(torch.abs(out - golden))
        print("Loop: ",i)
        if(not torch.allclose(out,golden,0.5,0.5)):
            print("Op name: ", ttf_binary_functions[i].__name__)
            embed()
        assert torch.allclose(out,golden,0.5,0.5), "Maximum diffeerence (%d)" % max_diff

    for i in range(len(ttf_unary_functions)):
        print("Op name: ", ttf_unary_functions[i].__name__)
        out_ttens = ttf_unary_functions[i](lin_ttens, op_dtype, runtime, fold_factors)
        out = out_ttens.from_device(0)
        out = out.type(torch.float32)
        golden = ttf_unary_functions[i](lin)
        max_diff = torch.max(torch.abs(out - golden))
        print("Loop: ",i)
        if(not torch.allclose(out,golden,0.5,0.5)):
            print("Op name: ", ttf_unary_functions[i].__name__)
            embed()
        assert torch.allclose(out,golden,0.5,0.5), "Maximum diffeerence (%d)" % max_diff
        del(out_ttens)

    for i in range(len(ttf_reduction_functions)):
        print("Op name: ", ttf_reduction_functions[i].__name__)
        dim = -1
        out_ttens = ttf_reduction_functions[i](rin_ttens, dim=dim, op_dtype=op_dtype, runtime=runtime)
        out = out_ttens.from_device(0)
        out = out.type(torch.float32)
        if(dim == -1):
            pass
        else:
            golden = ttf_reduction_functions[i](rin,dim)
        embed()
        max_diff = torch.max(torch.abs(out - golden))
        print("Loop: ",i)
        if(not torch.allclose(out,golden,0.5,0.5)):
            print("Op name: ", ttf_reduction_functions[i].__name__)
            embed()
        assert torch.allclose(out,golden,0.5,0.5), "Maximum difference (%d)" % max_diff
        del(out_ttens)

    print("Passed TT functional test")

    ##
    # Delete TT tensors so that all allocated DRAM is put back, and shut down
    ##
    del(lin_ttens)
    del(rin_ttens)
    del(linmm_ttens)
    del(rinmm_ttens)
    runtime.simd_cluster.check_allocator_end_state()

def set_vertical_zero_stripe(tens):
    out = tens.clone()
    for j in range(128):
        for i in range(64):
            out[0][0][0][j][i] = 0
        for i in range(64):
            out[0][0][0][j][i+64] = tens[0][0][0][j][i]
    return out

def set_horizontal_zero_stripe(tens):
    out = tens.clone()
    for j in range(128):
        for i in range(64):
            out[0][0][0][i][j] = tens[0][0][0][i][j]
        for i in range(64):
            out[0][0][0][i+64][j] = 0
    return out

def transpose_test(simd0, netlist, runtime, backend, be_api):
    block_size = 64
    simd0.set_up_allocators([(tt_dtype.Float16, 64, 10000, 0x21000000)])
    simd0.set_up_allocators([(tt_dtype.Float32, 64, 10000, 0x31000000)])

    lin = torch.randn(1,1,1,128,256)
    rin = torch.randn(1,1,1,128,256)

    lin_ttens = tt_tensor(block_size=block_size, simd_cluster=runtime.simd_cluster, torch_tensor=lin, dtype=tt_dtype.Float32)
    rin_ttens = tt_tensor(block_size=block_size, simd_cluster=runtime.simd_cluster, torch_tensor=rin, dtype=tt_dtype.Float32)
    lin_ttens.to_device(0,lin)
    rin_ttens.to_device(0,rin)
    transposed = rin_ttens.transpose()
    row_fold = random.choice([1,2,4])
    col_fold = random.choice([1,2,4])
    out_ttens = ttf.matmul(lin_ttens, transposed, tt_op_dtype(tt_dtype.Float16), runtime, (2,2,2))
    out = out_ttens.from_device(0)
    out = out.type(torch.float32)
    golden = torch.matmul(lin,torch.transpose(rin,-1,-2))
    assert torch.allclose(out,golden,atol=1.8,rtol=1.8), "Maximum difference"

def test_softmax(simd0, netlist,runtime, backend, be_api):
    lin, rin, linmm, rinmm, block_size, fold_factors = gen_random_inputs()
    simd0.set_up_allocators([(tt_dtype.Float32, block_size, 10000, 0x11000000)])
    simd0.set_up_allocators([(tt_dtype.Float16, block_size, 10000, 0x21000000)])
    simd0.set_up_allocators([(tt_dtype.Float16_b, block_size, 10000, 0x31000000)])

    lin_ttens = tt_tensor(block_size=block_size, simd_cluster=runtime.simd_cluster, torch_tensor=lin, dtype=tt_dtype.Float32)
    lin_ttens.to_device(0,lin)

    tout = ttf.softmax(lin_ttens,-1,tt_op_dtype(dtype=tt_dtype.Float16_b,dtype_intermed=tt_dtype.Float16_b, dtype_accum=tt_dtype.Float16_b),runtime=runtime,fold_factors=fold_factors)
    out = tout.from_device(0)
    outfloat = out.type(torch.float32)

    texp = torch.exp(lin)
    tred = texp.sum(-1)
    trecip = 1/tred
    trecip_for_use = trecip.unsqueeze(-1).broadcast_to(1,1,1,outfloat.shape[-2],outfloat.shape[-1])
    tres = texp * trecip_for_use

    assert torch.allclose(tres,outfloat,atol=0.1,rtol=0.001)

def main():
    print("Testing TT functional!")
    target_arch = BackendDevice.Grayskull
    target_devices = {0}
    config = be_api.get_runtime_config(target_arch)
    backend = Backend(config, target_devices)
    be_api.initialize_child_process(target_arch, target_devices)
    netlist = tt_netlist()
    simd0 = tt_simd_cluster(4,8, list(range(4*8)), be_api=be_api, netlist=netlist)
    runtime = tt_runtime(simd0, netlist, be_api, backend)

    for x in range(5):
        test_softmax(simd0, netlist,runtime, backend, be_api)
        test_ops(simd0, netlist,runtime, backend, be_api)
        transpose_test(simd0, netlist,runtime, backend, be_api)

    print("Finished testing TT functional!")

    be_api.finish_child_process()
    backend.destroy()
    # print("Successfully done test")
if __name__ == "__main__":
    main()




