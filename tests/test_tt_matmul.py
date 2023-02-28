import torch
import random
import logging
from tt_dtype import block_size_bytes, tt_dtype, tt_op_dtype
from tt_simd_cluster import tt_simd_cluster
from tt_tensor import tt_tensor
import time
import numpy as np
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
from argparse import ArgumentParser

def mean_absolute_fraction_error(exp, rec):
    mae = torch.mean(torch.abs((exp - rec)))
    return mae / torch.mean(torch.abs(exp))


def test_matmul(target_arch):
    '''
    What does a test need?
    tt_simd_cluster w/ allocators set up
    '''
    # Why is the user creating all of these?
    simd = tt_simd_cluster(1, 1, [0,], be_api, arch=target_arch)
    target_devices = {0}
    config = be_api.get_runtime_config(target_arch)

    backend = Backend(config, target_devices)
    be_api.initialize_child_process(target_arch, target_devices) # Why is user launching child process?
    netlist = tt_netlist(target_arch)
    runtime = tt_runtime(simd, netlist, be_api, backend) # Why is the runtime a thing?
    dtype = tt_dtype.Float16
    op_dtype = tt_op_dtype(dtype, dtype_intermed=dtype, dtype_accum=dtype)
    block_size = 64
    simd.set_up_allocators([(dtype, block_size, 2000, 250000000)])
    simd.netlist = netlist

    dims = 64, 512, 512
    shape0 = (1, dims[0], dims[1])
    shape1 = (1, dims[1], dims[2])
    A = torch.randn(shape0)
    B = torch.randn(shape1)
    golden = torch.matmul(A, B)

    logging.info("Creating empty tt_tensors")

    tt_A = tt_tensor(block_size, simd, torch_tensor=A, dtype=dtype)
    tt_B = tt_tensor(block_size, simd, torch_tensor=B, dtype=dtype)

    logging.info("Pushing data to device RAM")
    tt_A.to_device(0, A)
    tt_B.to_device(0, B)

    logging.info("Running ttf.matmul")
    tt_out = ttf.matmul(tt_A, tt_B, op_dtype=op_dtype, runtime=runtime)

    logging.info("Ran ttf.matmul. Getting output from device")
    out = tt_out.from_device(0)

    logging.info("Received output from device")
    mse = torch.mean((out - golden)**2)
    logging.info(f"Mean Squared Error: {mse}")

    logging.info(f'Expected: {golden}')
    logging.info(f'Recieved: {out}')

    # A_from = tt_A.from_device(0)
    # B_from = tt_B.from_device(0)

    # assert torch.allclose(A, A_from)
    # assert torch.allclose(B, B_from)
    # print('Test passed: SUCCESS')

    # Why is user doing this cleanup??
    be_api.finish_child_process()
    backend.destroy()

def test_matmul_2xchip(target_arch):
    '''
    A @ B @ C on a 1x2 cluster. B and C sharded by columns
    '''
    simd = tt_simd_cluster(1, 2, [0,], be_api, arch=target_arch)
    target_devices = {0, 1}
    config = be_api.get_runtime_config(target_arch)
    backend = Backend(config, target_devices)
    be_api.initialize_child_process(target_arch, target_devices)
    netlist = tt_netlist(target_arch)
    runtime = tt_runtime(simd, netlist, be_api, backend)
    dtype = tt_dtype.Float16
    op_dtype = tt_op_dtype(dtype, dtype_intermed=dtype, dtype_accum=dtype)
    block_size = 128
    simd.set_up_allocators([(dtype, block_size, 2000, 250000000)])
    simd.netlist = netlist
    simd.runtime = runtime

    dims = 256, 1024, 1024, 1024
    shape0 = (1, dims[0], dims[1])
    shape1 = (1, dims[1], dims[2])
    shape2 = (1, dims[2], dims[3])
    A = torch.randn(shape0)
    B = torch.randn(shape1)
    C = torch.randn(shape2)
    hidden = torch.matmul(A, B)
    golden = torch.matmul(hidden, C)

    logging.info("Creating empty tt_tensors")

    tt_A = tt_tensor(block_size, simd, torch_tensor=A, dtype=dtype)
    tt_B = tt_tensor(block_size, simd, torch_tensor=B, dtype=dtype)
    tt_C = tt_tensor(block_size, simd, torch_tensor=C, dtype=dtype)

    logging.info("Pushing data to device RAM")
    tt_A.to_device(0, A, 1, 4) # n0
    tt_B.to_device(0, B, 1, 8) # n1
    tt_C.to_device(0, C, 1, 8) # n2

    logging.info("Sharding tt_B")
    sharded_B = tt_B.shard(-2, -1) # n3

    logging.info("Sharding tt_C")
    sharded_C = tt_C.shard(-2, -1)

    logging.info("Running ttf.matmul A*B")
    tt_out_0 = ttf.matmul(tt_A, sharded_B, op_dtype=op_dtype, runtime=runtime)

    logging.info("Ran ttf.matmul A*B. Unsharding output")
    output_unsharded_0 = tt_out_0.unshard(-2, -1)

    logging.info("Running ttf.matmul hidden*C")
    tt_out_1 = ttf.matmul(output_unsharded_0, sharded_C, op_dtype=op_dtype, runtime=runtime)

    logging.info("Ran ttf.matmul hidden*C. Unsharding output")
    output_unsharded_1 = tt_out_1.unshard(-2, -1)

    logging.info("Getting output from device")
    out = output_unsharded_1.from_device(0)

    logging.info("Received output from device")
    mse = torch.mean((out - golden)**2)
    logging.info(f"Mean Squared Error: {mse}")

    logging.debug(f"golden.shape:{golden.shape}")
    logging.debug(f"out.shape:{out.shape}")

    logging.info(f'Expected: {golden}')
    logging.info(f'Received: {out}')

    # A_from = tt_A.from_device(0)
    # B_from = tt_B.from_device(0)

    # assert torch.allclose(A, A_from)
    # assert torch.allclose(B, B_from)
    # print('Test passed: SUCCESS')

    # Why is user doing this cleanup??
    be_api.finish_child_process()
    backend.destroy()

def test_matmul_2xchip_1d_weight_stationary(target_arch):
    '''
    A @ B @ C on a 1x2 cluster. B sharded by columns, C sharded by rows
    '''
    simd = tt_simd_cluster(1, 2, [0,], be_api, arch=target_arch)
    target_devices = {0, 1}
    config = be_api.get_runtime_config(target_arch)
    backend = Backend(config, target_devices)
    be_api.initialize_child_process(target_arch, target_devices) # Why is user launching child process?
    netlist = tt_netlist(target_arch)
    runtime = tt_runtime(simd, netlist, be_api, backend) # Why is the runtime a thing?
    dtype = tt_dtype.Float16
    op_dtype = tt_op_dtype(dtype, dtype_intermed=dtype, dtype_accum=dtype)
    block_size = 128
    simd.set_up_allocators([(dtype, block_size, 2000, 250000000)])
    simd.netlist = netlist
    simd.runtime = runtime

    dims = 256, 512, 512, 256
    shape0 = (1, dims[0], dims[1])
    shape1 = (1, dims[1], dims[2])
    shape2 = (1, dims[2], dims[3])
    A = torch.randn(shape0)
    B = torch.randn(shape1)
    C = torch.randn(shape2)
    hidden = torch.matmul(A, B)
    golden = torch.matmul(hidden, C)

    logging.info("Creating empty tt_tensors")

    tt_A = tt_tensor(block_size, simd, torch_tensor=A, dtype=dtype)
    tt_B = tt_tensor(block_size, simd, torch_tensor=B, dtype=dtype)
    tt_C = tt_tensor(block_size, simd, torch_tensor=C, dtype=dtype)

    logging.info("Pushing data to device RAM")
    tt_A.to_device(0, A)
    tt_B.to_device(0, B)
    tt_C.to_device(0, C)

    logging.info("Sharding tt_B")
    sharded_B = tt_B.shard(rsplit_dim=-2, csplit_dim=-1)

    logging.info("Sharding tt_C")
    sharded_C = tt_C.shard(rsplit_dim=-2, csplit_dim=-2)

    logging.info("Running ttf.matmul A*B")
    tt_out_0 = ttf.matmul(tt_A, sharded_B, op_dtype=op_dtype, runtime=runtime)

    logging.info("Ran ttf.matmul A*B. Leave output sharded in dim=-1")

    logging.info("Running ttf.matmul hidden*C")
    tt_out_1 = ttf.matmul(tt_out_0, sharded_C, op_dtype=op_dtype, runtime=runtime)

    logging.info("Ran ttf.matmul hidden*C. Unsharding output")
    output_unsharded_1 = tt_out_1.unshard(rsplit_dim=-2, csplit_dim=-1)

    logging.info("unsharded output contains the partial products stacked in the column dimension")
    output_shape = (1, dims[0]//block_size, dims[3]//block_size)
    logging.info("Defining a tensor for each partial product")
    output_A = tt_tensor(block_size, simd, shape=output_shape, addr_tensor=output_unsharded_1.address_tensor[:,:,:2], parent_tensor=output_unsharded_1)
    output_B = tt_tensor(block_size, simd, shape=output_shape, addr_tensor=output_unsharded_1.address_tensor[:,:,2:], parent_tensor=output_unsharded_1)
    logging.info("Reducing the partial products into final output")
    output_reduced = ttf.add(output_A, output_B, op_dtype=op_dtype, runtime=runtime)

    logging.info("Getting output from device")
    out = output_reduced.from_device(0)

    logging.info("Received output from device")
    mse = torch.mean((out - golden)**2)
    logging.info(f"Mean Squared Error: {mse}")

    logging.debug(f"golden.shape:{golden.shape}")
    logging.debug(f"out.shape:{out.shape}")

    logging.info(f'Expected: {golden}')
    logging.info(f'Received: {out}')

    # A_from = tt_A.from_device(0)
    # B_from = tt_B.from_device(0)

    # assert torch.allclose(A, A_from)
    # assert torch.allclose(B, B_from)
    # print('Test passed: SUCCESS')

    # Why is user doing this cleanup??
    be_api.finish_child_process()
    backend.destroy()

def test_matmul_gelu(target_arch):
    '''
    What does a test need?
    tt_simd_cluster w/ allocators set up
    '''
    # Why is the user creating all of these?
    simd = tt_simd_cluster(1, 1, [0,], be_api, arch=target_arch)
    target_devices = {0}
    config = be_api.get_runtime_config(target_arch)
    backend = Backend(config, target_devices)
    be_api.initialize_child_process(target_arch, target_devices) # Why is user launching child process?
    netlist = tt_netlist(target_arch)
    runtime = tt_runtime(simd, netlist, be_api, backend) # Why is the runtime a thing?
    dtype = tt_dtype.Float32
    op_dtype = tt_op_dtype(dtype, dtype_intermed=dtype, dtype_accum=dtype)
    block_size = 128
    simd.set_up_allocators([(dtype, block_size, 2000, 250000000)])

    shape = (1, 1024, 1024)
    A = torch.randn(shape)
    B = torch.randn(shape)
    golden = torch.nn.functional.gelu(torch.matmul(A, B))

    logging.info("Creating empty tt_tensors")

    tt_A = tt_tensor(block_size, simd, torch_tensor=A, dtype=dtype)
    tt_B = tt_tensor(block_size, simd, torch_tensor=B, dtype=dtype)

    logging.info("Pushing data to device RAM")
    tt_A.to_device(0, A)
    tt_B.to_device(0, B)

    logging.info("Running ttf.matmul")
    tt_out = ttf.matmul(tt_A, tt_B, op_dtype=op_dtype, runtime=runtime)

    tt_gelu_out = ttf.gelu(tt_out, op_dtype=op_dtype, runtime=runtime)

    logging.info("Ran ttf.matmul. Getting output from device")
    out = tt_gelu_out.from_device(0)

    logging.info("Received output from device")
    mse = torch.mean((out - golden)**2)
    logging.info(f"Mean Squared Error: {mse}")

    logging.info(f'Expected: {golden}')
    logging.info(f'Recieved: {out}')

    # A_from = tt_A.from_device(0)
    # B_from = tt_B.from_device(0)

    # assert torch.allclose(A, A_from)
    # assert torch.allclose(B, B_from)
    # print('Test passed: SUCCESS')

    # Why is user doing this cleanup??
    be_api.finish_child_process()
    backend.destroy()

def test_matmul_gelu_matmul(target_arch):
    '''
    What does a test need?
    tt_simd_cluster w/ allocators set up
    '''
    # Why is the user creating all of these?
    simd = tt_simd_cluster(1, 1, [0,], be_api, arch=target_arch)
    target_devices = {0}
    config = be_api.get_runtime_config(target_arch)
    backend = Backend(config, target_devices)
    be_api.initialize_child_process(target_arch, target_devices) # Why is user launching child process?
    netlist = tt_netlist(target_arch)
    runtime = tt_runtime(simd, netlist, be_api, backend) # Why is the runtime a thing?
    dtype = tt_dtype.Float32
    op_dtype = tt_op_dtype(dtype, dtype_intermed=dtype, dtype_accum=dtype)
    block_size = 128
    simd.set_up_allocators([(dtype, block_size, 2000, 250000000)])

    shape = (1, 1024, 1024)
    A = torch.randn(shape)
    B = torch.randn(shape)
    C = torch.randn(shape)
    golden = torch.matmul(torch.nn.functional.gelu(torch.matmul(A, B)), C)

    logging.info("Creating empty tt_tensors")

    tt_A = tt_tensor(block_size, simd, torch_tensor=A, dtype=dtype)
    tt_B = tt_tensor(block_size, simd, torch_tensor=B, dtype=dtype)
    tt_C = tt_tensor(block_size, simd, torch_tensor=C, dtype=dtype)

    logging.info("Pushing data to device RAM")
    tt_A.to_device(0, A)
    tt_B.to_device(0, B)
    tt_C.to_device(0, C)

    logging.info("Running ttf.matmul")
    tt_ff1_out = ttf.matmul(tt_A, tt_B, op_dtype=op_dtype, runtime=runtime)

    tt_gelu_out = ttf.gelu(tt_ff1_out, op_dtype=op_dtype, runtime=runtime)

    tt_ff2_out = ttf.matmul(tt_gelu_out, tt_C, op_dtype=op_dtype, runtime=runtime)

    logging.info("Ran ttf.matmul. Getting output from device")
    out = tt_ff2_out.from_device(0)

    logging.info("Received output from device")
    logging.info(f"MAFE: {mean_absolute_fraction_error(golden, out)}")

    logging.info(f'Expected: {golden}')
    logging.info(f'Recieved: {out}')

    # A_from = tt_A.from_device(0)
    # B_from = tt_B.from_device(0)

    # assert torch.allclose(A, A_from)
    # assert torch.allclose(B, B_from)
    # print('Test passed: SUCCESS')

    # Why is user doing this cleanup??
    be_api.finish_child_process()
    backend.destroy()


def test_attn(target_arch):
    '''
    Functional attention module to test on WH
    '''
    simd = tt_simd_cluster(1, 1, [0,], be_api, arch=target_arch)
    target_devices = {0}
    config = be_api.get_runtime_config(target_arch)
    backend = Backend(config, target_devices)
    be_api.initialize_child_process(target_arch, target_devices) # Why is user launching child process?
    netlist = tt_netlist(target_arch)
    runtime = tt_runtime(simd, netlist, be_api, backend) # Why is the runtime a thing?
    dtype = tt_dtype.Float32
    op_dtype = tt_op_dtype(dtype, dtype_intermed=dtype, dtype_accum=dtype)
    block_size = 128 # can't be bigger than dhead, can't be so small we require 16x16 cores
    simd.set_up_allocators([(dtype, block_size, 2000, 250000000)])

    '''
    Q, K, V projections
    '''

    s, dm, dh, nh = 256, 1024, 1024//4, 4
    x = torch.randn(1, s, dm)
    Wq, Wv, Wk = (torch.randn(1, dm, dm) for _ in range(3))
    q = torch.matmul(x, Wq) # Ignore bias and weight tranposing
    k = torch.matmul(x, Wk)
    v = torch.matmul(x, Wv)

    logging.info("Pushing weights to device")
    tt_x = tt_tensor(block_size, simd, torch_tensor=x, dtype=dtype)
    logging.info("pushed x to device")
    tt_Wq = tt_tensor(block_size, simd, torch_tensor=Wq, dtype=dtype)
    logging.info("pushed Wq to device")
    tt_Wk = tt_tensor(block_size, simd, torch_tensor=Wk, dtype=dtype)
    logging.info("pushed Wk to device")
    tt_Wv = tt_tensor(block_size, simd, torch_tensor=Wv, dtype=dtype)
    logging.info("pushed Wv to device")

    tt_x.to_device(0, x)

    tt_Wq.to_device(0, Wq)
    tt_Wk.to_device(0, Wk)
    tt_Wv.to_device(0, Wv)

    tt_q = ttf.matmul(tt_x, tt_Wq, op_dtype=op_dtype, runtime=runtime)
    tt_k = ttf.matmul(tt_x, tt_Wk, op_dtype=op_dtype, runtime=runtime)
    tt_v = ttf.matmul(tt_x, tt_Wv, op_dtype=op_dtype, runtime=runtime)

    tt_q_cpu = tt_q.from_device(0)
    tt_k_cpu = tt_k.from_device(0)
    tt_v_cpu = tt_v.from_device(0)

    for exp, rec in zip([q, k, v], [tt_q_cpu, tt_k_cpu, tt_v_cpu]):
        logging.info('MAFE: ', mean_absolute_fraction_error(exp, rec))


    '''
    Q * K.T attention scores
    '''
    # s, d -> nh, s, dm
    q = q.reshape(s, nh, dh).permute(1,0,2)
    k = k.reshape(s, nh, dh).permute(1,2,0)
    scores = torch.matmul(q, k)
    

    # reshape in terms of blocks
    tt_q = tt_q.reshape((s//block_size, nh, dm//block_size//nh)).swapaxes(-2, -3)
    print('tt_q shape after reshape:', tt_q.shape)
    
    tt_k = tt_k.reshape((s//block_size, nh, dm//block_size//nh)).swapaxes(-2, -3).transpose()
    print('tt_k shape after reshape:', tt_k.shape)

    tt_scores = ttf.matmul(tt_q, tt_k, op_dtype=op_dtype, runtime=runtime)
    print('tt_scores shape: ', tt_scores.shape)

    tt_scores_cpu = tt_scores.from_device(0)
    print('expected scores shape:', scores.size())
    print(f'tt_scores shape:', tt_scores_cpu.size())
    
    logging.info(f"Scores MAFE: {mean_absolute_fraction_error(scores, tt_scores_cpu)*100:.3f}%")

    probs = torch.nn.functional.softmax(scores, dim=-1)

    # # We don't yet have stable softmax because reduce_max not supported. Do it from pytorch for now.
    scores_max = torch.max(tt_scores_cpu, dim=-1, keepdim=True)[0].expand(tt_scores_cpu.size()) # TODO: do on device
    tt_scores_max = tt_tensor(block_size, simd, torch_tensor=scores_max, dtype=dtype).to_device(0, scores_max)
    tt_scores_normed = ttf.subtract(tt_scores, tt_scores_max, op_dtype=op_dtype, runtime=runtime)

    tt_probs = ttf.softmax(tt_scores_normed, dim=-1, op_dtype=op_dtype, runtime=runtime, fold_factors=(1,1,1))
    tt_probs_cpu = tt_probs.from_device(0)
    print('reduction:', tt_probs_cpu)
    logging.info(f'Probabilities MAFE: {mean_absolute_fraction_error(probs, tt_probs_cpu):.3f}')


    '''
    probs * V
    '''
    v = v.reshape(s, nh, dh).permute(1,0,2)
    attn_out = torch.matmul(probs, v)

    tt_v = tt_v.reshape((s//block_size, nh, dm//block_size//nh)).swapaxes(-2, -3)
    tt_attn_out = ttf.matmul(tt_probs, tt_v, op_dtype=op_dtype, runtime=runtime)
    tt_attn_out_cpu = tt_attn_out.from_device(0)

    print('attn out expected:', attn_out.size(), attn_out)
    print('attn out received:',tt_attn_out_cpu.size(), tt_attn_out_cpu)
    print('attn out MAFE:', mean_absolute_fraction_error(attn_out, tt_attn_out_cpu))

    
    be_api.finish_child_process()
    backend.destroy()


def test_softmax(target_arch):
    simd = tt_simd_cluster(1, 1, [0,], be_api, arch=target_arch)
    target_devices = {0}
    config = be_api.get_runtime_config(target_arch)
    backend = Backend(config, target_devices)
    be_api.initialize_child_process(target_arch, target_devices) # Why is user launching child process?
    netlist = tt_netlist(target_arch)
    runtime = tt_runtime(simd, netlist, be_api, backend) # Why is the runtime a thing?
    dtype = tt_dtype.Float32
    op_dtype = tt_op_dtype(dtype, dtype_intermed=dtype, dtype_accum=dtype)
    block_size = 128
    simd.set_up_allocators([(dtype, block_size, 2000, 250000000)])

    shape = (1, 5, block_size*8, block_size*4)
    inp = torch.randn(shape)
    exp = torch.nn.functional.softmax(inp, dim=-1)

    tt_inp = tt_tensor(block_size, simd, torch_tensor=inp, dtype=dtype)
    tt_inp.to_device(0, inp)
    tt_out = ttf.softmax(tt_inp, dim=-1, op_dtype=op_dtype, runtime=runtime, fold_factors=(1,1,1))
    tt_out_cpu = tt_out.from_device(0)

    print('expected size:', exp.size())
    print(exp)
    print('received shape:', tt_out_cpu.shape)
    print(tt_out_cpu)

    print('MAFE:', mean_absolute_fraction_error(exp, tt_out_cpu))

    be_api.finish_child_process()
    backend.destroy()

def test_layernorm(target_arch):
    simd = tt_simd_cluster(1, 1, [0,], be_api, arch=target_arch)
    target_devices = {0}
    config = be_api.get_runtime_config(target_arch)
    backend = Backend(config, target_devices)
    be_api.initialize_child_process(target_arch, target_devices) # Why is user launching child process?
    netlist = tt_netlist(target_arch)
    runtime = tt_runtime(simd, netlist, be_api, backend) # Why is the runtime a thing?
    dtype = tt_dtype.Float32
    op_dtype = tt_op_dtype(dtype, dtype_intermed=dtype, dtype_accum=dtype)
    block_size = 128
    simd.set_up_allocators([(dtype, block_size, 2000, 250000000)])

    # import pdb
    # pdb.set_trace()
    shape = (1, 5, block_size*8, block_size*4)
    inp = torch.randn(shape)
    normalized_shape = shape[-1]
    gamma = torch.randn(normalized_shape)
    beta = torch.randn(normalized_shape)
    exp = torch.nn.functional.layer_norm(inp, gamma.size(), weight=gamma, bias=beta, eps=1e-05)

    tt_inp = tt_tensor(block_size, simd, torch_tensor=inp, dtype=dtype).to_device(0, inp)
    gamma_expand = gamma.expand(inp.size()) # I think we need to make this the same size as input for tt_functional
    tt_gamma = tt_tensor(block_size, simd, torch_tensor=gamma_expand, dtype=dtype).to_device(0, gamma_expand)
    beta_expand = beta.expand(inp.size())
    tt_beta = tt_tensor(block_size, simd, torch_tensor=beta_expand, dtype=dtype).to_device(0, beta_expand)

    tt_out = ttf.layer_norm(tt_inp, tt_beta, tt_gamma, op_dtype=op_dtype, runtime=runtime, fold_factors=(1,1,1))
    tt_out_cpu = tt_out.from_device(0)

    print('expected size:', exp.size())
    print(exp)
    print('received shape:', tt_out_cpu.shape)
    print(tt_out_cpu)

    print('MAFE:', mean_absolute_fraction_error(exp, tt_out_cpu))

    be_api.finish_child_process()
    backend.destroy()

def test_layernorm_2xchip(target_arch):
    '''
    Input on a 2x1 cluster, sharded by rows
    '''
    simd = tt_simd_cluster(2, 1, [0,], be_api, arch=target_arch)
    target_devices = set(simd.get_chip_ids())
    config = be_api.get_runtime_config(target_arch)
    backend = Backend(config, target_devices)
    be_api.initialize_child_process(target_arch, target_devices)
    netlist = tt_netlist(target_arch)
    runtime = tt_runtime(simd, netlist, be_api, backend)
    dtype = tt_dtype.Float32
    op_dtype = tt_op_dtype(dtype, dtype_intermed=dtype, dtype_accum=dtype)
    block_size = 256
    simd.set_up_allocators([(dtype, block_size, 2000, 250000000)])
    simd.netlist = netlist
    simd.runtime = runtime

    shape = (1, 4, 2048, 512)
    A = torch.randn(shape)  # activation for layernorm
    normalized_shape = shape[-1]
    gamma = torch.randn(normalized_shape)
    beta = torch.randn(normalized_shape)
    golden = torch.nn.functional.layer_norm(A, gamma.size(), weight=gamma, bias=beta, eps=1e-05)

    logging.info("Pushing tt_A to device")
    tt_A = tt_tensor(block_size, simd, torch_tensor=A, dtype=dtype).to_device(0, A)

    logging.info("Sharding tt_A")
    sharded_A = tt_A.shard(-2, -1)

    logging.info("Expanding gamma and beta to match sharded_A")
    gamma_expand = gamma.expand(sharded_A.size())
    beta_expand = beta.expand(sharded_A.size())
    tt_gamma = tt_tensor(block_size, simd, torch_tensor=gamma_expand, dtype=dtype).to_device(0, gamma_expand)
    tt_beta = tt_tensor(block_size, simd, torch_tensor=beta_expand, dtype=dtype).to_device(0, beta_expand)

    logging.info("Running LayerNorm")
    tt_out = ttf.layer_norm(sharded_A, tt_beta, tt_gamma, op_dtype=op_dtype, runtime=runtime, fold_factors=(1,1,1))

    logging.info("Unsharding output")
    unsharded_out = tt_out.unshard(-2, -1)

    logging.info("Getting output from device")
    out = unsharded_out.from_device(0)

    mse = torch.mean((out - golden)**2)
    logging.info(f"Mean Squared Error: {mse}")

    be_api.finish_child_process()
    backend.destroy()

def test_reduce_max(target_arch):
    simd = tt_simd_cluster(0, 0, [0,], be_api, arch=target_arch)
    target_devices = {0}
    config = be_api.get_runtime_config(target_arch)
    backend = Backend(config, target_devices)
    be_api.initialize_child_process(target_arch, target_devices) # Why is user launching child process?
    netlist = tt_netlist(target_arch)
    runtime = tt_runtime(simd, netlist, be_api, backend) # Why is the runtime a thing?
    dtype = tt_dtype.Float32
    op_dtype = tt_op_dtype(dtype, dtype_intermed=dtype, dtype_accum=dtype)
    block_size = 128
    simd.set_up_allocators([(dtype, block_size, 2000, 250000000)])

    shape = (1, 1, 1, 5, block_size*8, block_size*4)
    inp = torch.randn(shape)
    exp, _ = torch.max(inp, dim=-1, keepdim=True)
    print('expected size:', exp.size())
    print(exp)

    tt_inp = tt_tensor(block_size, simd, torch_tensor=inp, dtype=dtype)
    tt_out = ttf.reduce_max(tt_inp, dim=-1, op_dtype=op_dtype, runtime=runtime)
    tt_out_cpu = tt_out.from_device(0)
    print(tt_out_cpu)

    be_api.finish_child_process()
    backend.destroy()

def main(target_arch, test):
    if test == "matmul":
        test_matmul(target_arch)
    if test == "matmul_2xchip":
        test_matmul_2xchip(target_arch)
    # test_matmul_gelu(target_arch)
    # test_matmul_gelu_matmul(target_arch)
    # test_attn(target_arch)
    # test_softmax(target_arch)
    if test == "layernorm":
        test_layernorm(target_arch)
    if test == "layernorm_2xchip":
        test_layernorm_2xchip(target_arch)
    if test == "matmul_1d":
        test_matmul_2xchip_1d_weight_stationary(target_arch)

if __name__ == '__main__':
    logging.basicConfig(level="INFO")
    parser = ArgumentParser()
    parser.add_argument('--device', '-d', help='Device: {wh, gs}', default='wh')
    parser.add_argument('--test', '-t', help='Sub-test name', default='layernorm')
    args = parser.parse_args()
    target_arch = {'gs': BackendDevice.Grayskull, 'wh': BackendDevice.Wormhole}[args.device]
    main(target_arch, args.test)
