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

def test_matmul():
    target_arch = BackendDevice.Grayskull
    target_devices = {0}
    config = be_api.get_runtime_config(target_arch)
    backend = Backend(config, target_devices)
    be_api.initialize_child_process(target_arch, target_devices)

    simd0 = tt_simd_cluster(4,8, list(range(4*8)), be_api)
    netlist = tt_netlist()
    runtime = tt_runtime(simd0, netlist)

    simd0.set_up_allocators([(tt_dtype.Float32, 64, 10000, 0x21000000)])
    simd0.set_up_allocators([(tt_dtype.Float16, 64, 10000, 0x31000000)])

    ## Meat
    lin = torch.randn((1,1,1,512,1024))
    rin = torch.randn((1,1,1,1024,256))

    binary_op_test(tt_net_op_types.matmul, lin, rin, block_size=64, runtime=runtime, backend=backend)

    simd0.check_allocator_end_state()
    simd0.be_api.finish_child_process()
    backend.destroy()

    logging.info("Passed tt.functional matmul test")

def binary_op_test(op, lin, rin, block_size, runtime, backend):
    lin_ttens = tt_tensor(block_size=64, simd_cluster=runtime.simd_cluster, torch_tensor=lin, dtype=tt_dtype.Float32)
    rin_ttens = tt_tensor(block_size=64, simd_cluster=runtime.simd_cluster, torch_tensor=rin, dtype=tt_dtype.Float32)
    lin_ttens.to_device(0,lin)
    rin_ttens.to_device(0,rin)

    genout = ttf.tt_binary_op(op,lin_ttens,rin_ttens,tt_op_dtype(tt_dtype.Float16),runtime) 
    status = backend.compile_and_run_netlist(runtime.netlist.get_last_netlist_name(), {})
    assert status == BackendStatusCode.Success
    backend.wait_for_idle()

    print ("Genout shape: ", genout.shape)
    out = genout.from_device(0)
    out = out.type(torch.float32)
    golden_out = torch.matmul(lin,rin)

    max_diff = torch.max(torch.abs(out - golden_out))
    print("Maximum difference: ", max_diff)
    # Check vs golden
    print(out)
    print(golden_out)
    assert torch.allclose(out,golden_out,0.5,0.5)

def main():
    print("Testing TT functional!")
    #test_self_attn()
    for x in range(1):
        test_matmul()

if __name__ == "__main__":
    main()