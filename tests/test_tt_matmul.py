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

def test_matmul(target_arch):
    '''
    What does a test need?
    tt_simd_cluster w/ allocators set up
    '''
    # Why is the user creating all of these?
    simd = tt_simd_cluster(0, 0, [0,], be_api, arch=target_arch)
    target_devices = {0}
    config = be_api.get_runtime_config(target_arch)
    backend = Backend(config, target_devices)
    be_api.initialize_child_process(target_arch, target_devices) # Why is user launching child process?
    netlist = tt_netlist(target_arch)
    runtime = tt_runtime(simd, netlist, be_api, backend) # Why is the runtime a thing?
    dtype = tt_dtype.Float32
    op_dtype = tt_op_dtype(dtype)
    block_size = 64
    simd.set_up_allocators([(dtype, block_size, 2000, 250000000)])

    dims = 64, 512, 512
    shape0 = (1, 1, 1, dims[0], dims[1])
    shape1 = (1, 1, 1, dims[1], dims[2])
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


def test_matmul_gelu(target_arch):
    '''
    What does a test need?
    tt_simd_cluster w/ allocators set up
    '''
    # Why is the user creating all of these?
    simd = tt_simd_cluster(0, 0, [0,], be_api, arch=target_arch)
    target_devices = {0}
    config = be_api.get_runtime_config(target_arch)
    backend = Backend(config, target_devices)
    be_api.initialize_child_process(target_arch, target_devices, "./cluster_desc.yaml") # Why is user launching child process?
    netlist = tt_netlist(target_arch)
    runtime = tt_runtime(simd, netlist, be_api, backend) # Why is the runtime a thing?
    dtype = tt_dtype.Float32
    op_dtype = tt_op_dtype(dtype)
    block_size = 128
    simd.set_up_allocators([(dtype, block_size, 2000, 250000000)])

    shape = (1, 1, 1, 1024, 1024)
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
    simd = tt_simd_cluster(0, 0, [0,], be_api, arch=target_arch)
    target_devices = {0}
    config = be_api.get_runtime_config(target_arch)
    backend = Backend(config, target_devices)
    be_api.initialize_child_process(target_arch, target_devices) # Why is user launching child process?
    netlist = tt_netlist(target_arch)
    runtime = tt_runtime(simd, netlist, be_api, backend) # Why is the runtime a thing?
    dtype = tt_dtype.Float32
    op_dtype = tt_op_dtype(dtype)
    block_size = 128
    simd.set_up_allocators([(dtype, block_size, 2000, 250000000)])

    shape = (1, 1, 1, 1024, 1024)
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


def test_attn(target_arch):
    '''
    Functional attention module to test on WH
    '''
    simd = tt_simd_cluster(0, 0, [0,], be_api, arch=target_arch)
    target_devices = {0}
    config = be_api.get_runtime_config(target_arch)
    backend = Backend(config, target_devices)
    be_api.initialize_child_process(target_arch, target_devices) # Why is user launching child process?
    netlist = tt_netlist(target_arch)
    runtime = tt_runtime(simd, netlist, be_api, backend) # Why is the runtime a thing?
    dtype = tt_dtype.Float32
    op_dtype = tt_op_dtype(dtype)
    block_size = 128 # can't be bigger than dhead, can't be so small we require 16x16 cores
    simd.set_up_allocators([(dtype, block_size, 2000, 250000000)])

    '''
    Q, K, V projections
    '''

    s, dm, dh, nh = 256, 1024, 1024//4, 4
    x = torch.randn(1, 1, 1, s, dm)
    Wq, Wv, Wk = (torch.randn(1, 1, 1, dm, dm) for _ in range(3))
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
        mse = torch.mean((exp - rec)**2)
        logging.info(f"Mean Squared Error: {mse}")
    
    # exit()


    # '''
    # Q * K.T attention scores
    # '''
    # # s, d -> nh, s, dm
    # q = q.reshape(s, nh, dh).permute(1,0,2)
    # k = k.reshape(s, nh, dh).permute(1,2,0)
    # scores = torch.matmul(q, k)
    # print('expected scores shape:', scores.size())

    # print('tt_k address tensor:')
    # print(tt_k.address_tensor.size()) # torch.Size([1, 1, 1, 2, 8])
    # # reshape in terms of blocks
    # tt_scores = ttf.matmul(tt_q, tt_k, op_dtype=op_dtype, runtime=runtime, tms=[[{'hslice': [nh]}, 'tranpose'], [{'hslice': [nh]}]])
    # print('tt_scores shape:', tt_scores.shape)

    be_api.finish_child_process()
    backend.destroy()

def main(target_arch):
    # test_matmul_gelu_matmul(target_arch)
    # test_matmul(target_arch)
    test_attn(target_arch)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--device', '-d', help='Device: {wh, gs}', default='wh')
    args = parser.parse_args()
    target_arch = {'gs': BackendDevice.Grayskull, 'wh': BackendDevice.Wormhole}[args.device]
    main(target_arch)