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

def test_broadcast(target_arch):
    simd = tt_simd_cluster(0, 0, [0,], be_api, arch=target_arch)
    target_devices = {0,1}
    config = be_api.get_runtime_config(target_arch)
    backend = Backend(config, target_devices)
    be_api.initialize_child_process(target_arch, target_devices) # Why is user launching child process?
    netlist = tt_netlist(target_arch)
    runtime = tt_runtime(simd, netlist, be_api, backend) # Why is the runtime a thing?
    dtype = tt_dtype.Float32
    op_dtype = tt_op_dtype(dtype)
    block_size = 64
    simd.set_up_allocators([(dtype, block_size, 2000, 250000000)])

    shape = (1, 1, 1, 256, 256)
    A = torch.randn(shape)

    logging.info("Creating empty tt_tensor")

    tt_A = tt_tensor(block_size, simd, torch_tensor=A, dtype=dtype)

    logging.info("Pushing data to device RAM")
    tt_A.to_devices(A)

    output_shape_blocked = (1, 2, 1, 256 // block_size, 256 // block_size)
    output = tt_tensor(block_size=block_size, simd_cluster=simd, shape=output_shape_blocked, dtype=dtype)

    logging.info("Running ttf.broadcast")
    # returns a list of tt_tensors on each chip
    ttf.broadcast(tt_A, output, op_dtype=op_dtype, runtime=runtime)

    logging.info("Ran ttf.broadcast Getting tensors from device")

    out = output.from_devices()
    logging.info(f"Received output from devices {output.shape[0]}x{output.shape[1]}")

    # destroy bbe before checking errors, otherwise runtime does not clean up and next test hangs
    be_api.finish_child_process()
    backend.destroy()

    # compare the output from each chip
    for chip_r in range(output_shape_blocked[0]):
        for chip_c in range(output_shape_blocked[1]):
            device_out = out[chip_r][chip_c][:][:][:]
            print(f"device_out[{chip_r}][{chip_c}]={device_out}")
            assert torch.allclose(device_out, A, atol=1e-03, rtol=1e-02)
    print('Test passed: SUCCESS')

def main(target_arch):
    test_broadcast(target_arch)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--device', '-d', help='Device: {wh, gs}', default='wh')
    args = parser.parse_args()
    target_arch = {'gs': BackendDevice.Grayskull, 'wh': BackendDevice.Wormhole}[args.device]
    main(target_arch)
