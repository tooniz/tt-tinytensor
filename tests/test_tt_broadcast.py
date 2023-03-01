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

def test_broadcast(target_arch, num_chips):
    assert num_chips == 8 or num_chips == 2
    simd = tt_simd_cluster(1, num_chips, [0,], be_api, arch=target_arch)
    target_devices = {0, 3, 4, 7, 1, 2, 17, 8} if num_chips == 8 else {0, 1}
    config = be_api.get_runtime_config(target_arch)
    backend = Backend(config, target_devices)
    be_api.initialize_child_process(target_arch, target_devices) # Why is user launching child process?
    netlist = tt_netlist(target_arch)
    runtime = tt_runtime(simd, netlist, be_api, backend) # Why is the runtime a thing?
    dtype = tt_dtype.Float32
    op_dtype = tt_op_dtype(dtype)
    block_size = 128
    simd.set_up_allocators([(dtype, block_size, 2000, 250000000)])
    simd.netlist = netlist
    simd.runtime = runtime

    shape = (1, 1, 1, 256, block_size*num_chips)
    A = torch.randn(shape)

    logging.info("Creating input tt_tensor")

    tt_A = tt_tensor(block_size, simd, torch_tensor=A, dtype=dtype)

    logging.info("Pushing data to device RAM")

    tt_A.to_device(0, A)

    logging.info("Running shard")

    sharded_tensor = tt_A.shard(-2,-1)

    logging.info("Running unshard")

    output_unsharded = sharded_tensor.unshard(-2, -1)

    logging.info("Reading from device")

    out = output_unsharded.from_device(0)

    # destroy bbe before checking errors, otherwise runtime does not clean up and next test hangs
    be_api.finish_child_process()
    backend.destroy()

    assert torch.allclose(out, A[0][0], atol=1e-03, rtol=1e-02)
    print('Test passed: SUCCESS')

def main(target_arch, num_chips):
    test_broadcast(target_arch, num_chips)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--device', '-d', help='Device: {wh, gs}', default='wh')
    parser.add_argument('--num_chips', '-n', type=int, help='Num_chips: {2, 8}', default=2)
    args = parser.parse_args()
    target_arch = {'gs': BackendDevice.Grayskull, 'wh': BackendDevice.Wormhole}[args.device]
    main(target_arch, args.num_chips)
