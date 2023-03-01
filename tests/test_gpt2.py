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
    # TT setup
    simd = tt_simd_cluster(1, 1, [0,], be_api, arch=target_arch)
    target_devices = {0}
    config = be_api.get_runtime_config(target_arch)
    backend = Backend(config, target_devices)
    be_api.initialize_child_process(target_arch, target_devices)
    netlist = tt_netlist(target_arch)
    runtime = tt_runtime(simd, netlist, be_api, backend)
    dtype = tt_dtype.Float32
    op_dtype = tt_op_dtype(dtype, dtype_intermed=dtype, dtype_accum=dtype)
    block_size = 128
    simd.set_up_allocators([(dtype, block_size, 2000, 250000000)])

    '''
    LayerNorm
    '''
    # PyTorch
    s, dm, dh, nh = 256, 1024, 1024//4, 4
    x = torch.randn(1, s, dm)

    normalized_shape = dm
    gamma_ln1 = torch.randn(normalized_shape)
    beta_ln1 = torch.randn(normalized_shape)
    x_norm = torch.nn.functional.layer_norm(x, gamma_ln1.size(), weight=gamma_ln1, bias=beta_ln1, eps=1e-05)

    # TinyTensor
    tt_x = tt_tensor(block_size, simd, torch_tensor=x, dtype=dtype).to_device(0, x)
    gamma_ln1_expand = gamma_ln1.expand(x.size())
    tt_gamma_ln1 = tt_tensor(block_size, simd, torch_tensor=gamma_ln1_expand, dtype=dtype).to_device(0, gamma_ln1_expand)
    beta_ln1_expand = beta_ln1.expand(x.size())
    tt_beta_ln1 = tt_tensor(block_size, simd, torch_tensor=beta_ln1_expand, dtype=dtype).to_device(0, beta_ln1_expand)

    tt_x_norm = ttf.layer_norm(tt_x, beta=tt_beta_ln1, gamma=tt_gamma_ln1, op_dtype=op_dtype, runtime=runtime, fold_factors=(1,1,1))
    tt_x_norm_cpu = tt_x_norm.from_device(0)

    logging.info(f'LN1 MAE: {mean_absolute_fraction_error(x_norm, tt_x_norm_cpu):.3f}')

    '''
    Q, K, V projections
    '''
    # PyTorch
    Wq, Wv, Wk = (torch.randn(1, dm, dm) for _ in range(3))
    q = torch.matmul(x_norm, Wq)
    k = torch.matmul(x_norm, Wk)
    v = torch.matmul(x_norm, Wv)

    # TinyTensor
    tt_Wq = tt_tensor(block_size, simd, torch_tensor=Wq, dtype=dtype).to_device(0, Wq)
    tt_Wk = tt_tensor(block_size, simd, torch_tensor=Wk, dtype=dtype).to_device(0, Wk)
    tt_Wv = tt_tensor(block_size, simd, torch_tensor=Wv, dtype=dtype).to_device(0, Wv)

    tt_q = ttf.matmul(tt_x_norm, tt_Wq, op_dtype=op_dtype, runtime=runtime)
    tt_k = ttf.matmul(tt_x_norm, tt_Wk, op_dtype=op_dtype, runtime=runtime)
    tt_v = ttf.matmul(tt_x_norm, tt_Wv, op_dtype=op_dtype, runtime=runtime)

    tt_q_cpu = tt_q.from_device(0)
    tt_k_cpu = tt_k.from_device(0)
    tt_v_cpu = tt_v.from_device(0)

    for exp, rec in zip([q, k, v], [tt_q_cpu, tt_k_cpu, tt_v_cpu]):
        logging.info(f'MAE: {mean_absolute_fraction_error(exp, rec):.3f}')

    '''
    Q * K.T attention probabilities
    '''
    # PyTorch
    q = q.reshape(s, nh, dh).permute(1,0,2)
    k = k.reshape(s, nh, dh).permute(1,2,0)
    scores = torch.matmul(q, k)
    probs = torch.nn.functional.softmax(scores, dim=-1)
    
    # TinyTensor
    tt_q = tt_q.reshape((s//block_size, nh, dm//block_size//nh)).swapaxes(-2, -3)
    
    tt_k = tt_k.reshape((s//block_size, nh, dm//block_size//nh)).swapaxes(-2, -3).transpose()

    tt_scores = ttf.matmul(tt_q, tt_k, op_dtype=op_dtype, runtime=runtime)

    tt_scores_cpu = tt_scores.from_device(0)
    
    logging.info(f"Scores MAE: {mean_absolute_fraction_error(scores, tt_scores_cpu):.3f}")

    # Interoperability with pytorch when desired
    scores_max = torch.max(tt_scores_cpu, dim=-1, keepdim=True)[0].expand(tt_scores_cpu.size())
    tt_scores_max = tt_tensor(block_size, simd, torch_tensor=scores_max, dtype=dtype).to_device(0, scores_max)
    tt_scores_normed = ttf.subtract(tt_scores, tt_scores_max, op_dtype=op_dtype, runtime=runtime)

    tt_probs = ttf.softmax(tt_scores_normed, dim=-1, op_dtype=op_dtype, runtime=runtime, fold_factors=(1,1,1))
    tt_probs_cpu = tt_probs.from_device(0)

    logging.info(f'Probabilities MAE: {mean_absolute_fraction_error(probs, tt_probs_cpu):.3f}')

    '''
    attention_probabilities * V
    '''
    # PyTorch
    v = v.reshape(s, nh, dh).permute(1,0,2)
    attn_out = torch.matmul(probs, v)

    # TinyTensor
    tt_v = tt_v.reshape((s//block_size, nh, dm//block_size//nh)).swapaxes(-2, -3)
    tt_attn_out = ttf.matmul(tt_probs, tt_v, op_dtype=op_dtype, runtime=runtime)
    tt_attn_out_cpu = tt_attn_out.from_device(0)

    logging.info(f'Attention Out MAE: {mean_absolute_fraction_error(attn_out, tt_attn_out_cpu):.3f}')

    '''
    Dense Layer
    '''
    # PyTorch
    attn_out = attn_out.permute(1, 0, 2).reshape(1, s, dm)
    dense = torch.randn(1, dm, dm)
    dense_out = torch.matmul(attn_out, dense)

    # TinyTensor
    tt_dense = tt_tensor(block_size, simd, torch_tensor=dense, dtype=dtype).to_device(0, dense)
    tt_attn_out = tt_attn_out.swapaxes(-2, -3).reshape((1, s // block_size, dm // block_size))
    tt_dense_out = ttf.matmul(tt_attn_out, tt_dense, op_dtype=op_dtype, runtime=runtime)
    tt_dense_out_cpu = tt_dense_out.from_device(0)

    logging.info(f'Dense Out MAE: {mean_absolute_fraction_error(dense_out, tt_dense_out_cpu):.3f}')

    '''
    Residual & LayerNorm
    '''
    # PyTorch
    attn_res = torch.add(x, dense_out)

    gamma_ln2 = torch.randn(normalized_shape)
    beta_ln2 = torch.randn(normalized_shape)
    attn_res_norm = torch.nn.functional.layer_norm(attn_res, gamma_ln2.size(), weight=gamma_ln2, bias=beta_ln2, eps=1e-05)

    # TinyTensor
    gamma_ln2_expand = gamma_ln2.expand(x.size())
    tt_gamma_ln2 = tt_tensor(block_size, simd, torch_tensor=gamma_ln2_expand, dtype=dtype).to_device(0, gamma_ln2_expand)
    beta_ln2_expand = beta_ln2.expand(x.size())
    tt_beta_ln2 = tt_tensor(block_size, simd, torch_tensor=beta_ln2_expand, dtype=dtype).to_device(0, beta_ln2_expand)

    tt_attn_res = ttf.add(tt_x, tt_dense_out, op_dtype=op_dtype, runtime=runtime)
    tt_attn_res_norm = ttf.layer_norm(tt_attn_res, beta=tt_beta_ln2, gamma=tt_gamma_ln2, op_dtype=op_dtype, runtime=runtime, fold_factors=(1,1,1))
    tt_attn_res_norm_cpu = tt_attn_res_norm.from_device(0)

    logging.info(f'Residual Norm MAE: {mean_absolute_fraction_error(attn_res_norm, tt_attn_res_norm_cpu):.3f}')
    

    '''
    MLP & Residual
    '''
    # PyTorch
    ff1 = torch.randn(1, dm, 4*dm)
    ff2 = torch.randn(1, 4*dm, dm)

    ff1_out = torch.matmul(attn_res_norm, ff1)
    gelu_out = torch.nn.functional.gelu(ff1_out)
    ff2_out = torch.matmul(gelu_out, ff2)
    gpt2_out = torch.add(attn_res, ff2_out)

    # TinyTensor
    tt_ff1 = tt_tensor(block_size, simd, torch_tensor=ff1, dtype=dtype).to_device(0, ff1)
    tt_ff2 = tt_tensor(block_size, simd, torch_tensor=ff2, dtype=dtype).to_device(0, ff2)

    tt_ff1_out = ttf.matmul(tt_attn_res_norm, tt_ff1, op_dtype=op_dtype, runtime=runtime)
    tt_gelu_out = ttf.gelu(tt_ff1_out, op_dtype=op_dtype, runtime=runtime)
    tt_ff2_out = ttf.matmul(tt_gelu_out, tt_ff2, op_dtype=op_dtype, runtime=runtime)
    tt_gpt2_out = ttf.add(tt_attn_res, tt_ff2_out, op_dtype=op_dtype, runtime=runtime)

    tt_gpt2_out_cpu = tt_gpt2_out.from_device(0)

    logging.info(f'GPT2 decoder output MAE: {mean_absolute_fraction_error(gpt2_out, tt_gpt2_out_cpu):.3f}')


    be_api.finish_child_process()
    backend.destroy()


def main():
    logging.basicConfig(level="INFO")
    parser = ArgumentParser()
    parser.add_argument('--device', '-d', help='Device: {wh, gs}', default='wh')
    args = parser.parse_args()
    target_arch = {'gs': BackendDevice.Grayskull, 'wh': BackendDevice.Wormhole}[args.device]

    tt_gpt2(target_arch)

if __name__ == '__main__':
    main()