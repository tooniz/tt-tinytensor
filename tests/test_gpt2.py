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
from argparse import ArgumentParser


def mean_absolute_fraction_error(exp, rec):
    mae = torch.mean(torch.abs((exp - rec)))
    return mae / torch.mean(torch.abs(exp))

def tt_gpt2(target_arch):
    '''
    Functional GPT2 decoder 
    '''
    simd = tt_simd_cluster(1, 1, [0,], be_api, arch=target_arch)
    target_devices = {0}
    config = be_api.get_runtime_config(target_arch)
    backend = Backend(config, target_devices)
    be_api.initialize_child_process(target_arch, target_devices)
    netlist = tt_netlist(target_arch)
    runtime = tt_runtime(simd, netlist, be_api, backend)
    dtype = tt_dtype.Float32
    op_dtype = tt_op_dtype(dtype, dtype_intermed=dtype, dtype_accum=dtype)
    block_size = 128 # can't be bigger than dhead, can't be so small we require 16x16 cores
    simd.set_up_allocators([(dtype, block_size, 2000, 250000000)])

    '''
    Q, K, V projections
    '''

    # TODO: pre-norm X

    s, dm, dh, nh = 256, 1024, 1024//4, 4
    x = torch.randn(1, s, dm)
    Wq, Wv, Wk = (torch.randn(1, dm, dm) for _ in range(3))
    q = torch.matmul(x, Wq) # Ignore bias and weight tranposing
    k = torch.matmul(x, Wk)
    v = torch.matmul(x, Wv)

    # Pushing inputs to device
    tt_x = tt_tensor(block_size, simd, torch_tensor=x, dtype=dtype).to_device(0, x)
    tt_Wq = tt_tensor(block_size, simd, torch_tensor=Wq, dtype=dtype).to_device(0, Wq)
    tt_Wk = tt_tensor(block_size, simd, torch_tensor=Wk, dtype=dtype).to_device(0, Wk)
    tt_Wv = tt_tensor(block_size, simd, torch_tensor=Wv, dtype=dtype).to_device(0, Wv)

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
    probs = torch.nn.functional.softmax(scores, dim=-1)
    
    # reshape in terms of blocks
    tt_q = tt_q.reshape((s//block_size, nh, dm//block_size//nh)).swapaxes(-2, -3)
    
    tt_k = tt_k.reshape((s//block_size, nh, dm//block_size//nh)).swapaxes(-2, -3).transpose()

    tt_scores = ttf.matmul(tt_q, tt_k, op_dtype=op_dtype, runtime=runtime)

    tt_scores_cpu = tt_scores.from_device(0)
    
    logging.info(f"Scores MAFE: {mean_absolute_fraction_error(scores, tt_scores_cpu):.3f}%")

    # # We don't yet have stable softmax because reduce_max not supported. Do it from pytorch for now.
    scores_max = torch.max(tt_scores_cpu, dim=-1, keepdim=True)[0].expand(tt_scores_cpu.size()) # TODO: do on device
    tt_scores_max = tt_tensor(block_size, simd, torch_tensor=scores_max, dtype=dtype).to_device(0, scores_max)
    tt_scores_normed = ttf.subtract(tt_scores, tt_scores_max, op_dtype=op_dtype, runtime=runtime)

    tt_probs = ttf.softmax(tt_scores_normed, dim=-1, op_dtype=op_dtype, runtime=runtime, fold_factors=(1,1,1))
    tt_probs_cpu = tt_probs.from_device(0)
    logging.info(f'Probabilities MAFE: {mean_absolute_fraction_error(probs, tt_probs_cpu):.3f}')

    '''
    probs * V
    '''
    v = v.reshape(s, nh, dh).permute(1,0,2)
    attn_out = torch.matmul(probs, v)

    tt_v = tt_v.reshape((s//block_size, nh, dm//block_size//nh)).swapaxes(-2, -3)
    tt_attn_out = ttf.matmul(tt_probs, tt_v, op_dtype=op_dtype, runtime=runtime)
    tt_attn_out_cpu = tt_attn_out.from_device(0)

    print('attn out MAFE:', mean_absolute_fraction_error(attn_out, tt_attn_out_cpu))

    '''
    Reshape, dense layer
    nh, s, dh -> s, dm
    '''
    attn_out = attn_out.permute(1, 0, 2).reshape(1, s, dm)
    dense = torch.randn(1, dm, dm)
    dense_out = torch.matmul(attn_out, dense)

    tt_dense = tt_tensor(block_size, simd, torch_tensor=dense, dtype=dtype).to_device(0, dense)
    tt_attn_out = tt_attn_out.swapaxes(-2, -3).reshape((1, s // block_size, dm // block_size))
    tt_dense_out = ttf.matmul(tt_attn_out, tt_dense, op_dtype=op_dtype, runtime=runtime)

    print('dense out MAFE:', mean_absolute_fraction_error(dense_out, tt_dense_out.from_device(0)))

    '''
    MLP
    '''
    ff1 = torch.randn(1, dm, 4*dm)
    ff2 = torch.randn(1, 4*dm, dm)

    ff1_out = torch.matmul(dense_out, ff1)
    gelu_out = torch.nn.functional.gelu(ff1_out)
    ff2_out = torch.matmul(gelu_out, ff2)

    tt_ff1 = tt_tensor(block_size, simd, torch_tensor=ff1, dtype=dtype).to_device(0, ff1)
    tt_ff2 = tt_tensor(block_size, simd, torch_tensor=ff2, dtype=dtype).to_device(0, ff2)

    tt_ff1_out = ttf.matmul(tt_dense_out, tt_ff1, op_dtype=op_dtype, runtime=runtime)
    tt_gelu_out = ttf.gelu(tt_ff1_out, op_dtype=op_dtype, runtime=runtime)
    tt_ff2_out = ttf.matmul(tt_gelu_out, tt_ff2, op_dtype=op_dtype, runtime=runtime)

    tt_ff2_out_cpu = tt_ff2_out.from_device(0)

    print('attn out MAFE:', mean_absolute_fraction_error(ff2_out, tt_ff2_out_cpu))

    be_api.finish_child_process()
    backend.destroy()


def main():
    parser = ArgumentParser()
    parser.add_argument('--device', '-d', help='Device: {wh, gs}', default='wh')
    args = parser.parse_args()
    target_arch = {'gs': BackendDevice.Grayskull, 'wh': BackendDevice.Wormhole}[args.device]

    tt_gpt2(target_arch)

if __name__ == '__main__':
    main()