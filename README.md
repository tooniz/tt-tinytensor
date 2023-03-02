# TinyTensor

### TT Device ARCH
TinyTensor supports various Tenstorrent device architectures. To select between them, simply change `ARCH_NAME` environment variable:
- `export ARCH_NAME=grayskull`
- `export ARCH_NAME=wormhole`

### Installation

Create symlink `bbe` to point to your Buda backend repository (to be moved to third_party)

```bash
.
|-- bbe        # Buda Backend
|-- README.md  # This File
```

Set up the environment

```bash
export ARCH_NAME=wormhole
export ROOT=`git rev-parse --show-toplevel`
export BUDA_HOME=$ROOT/bbe
export PYTHONPATH=$ROOT:$ROOT/src:$ROOT/tests:$ROOT/bbe/build/obj/py_api:$ROOT/bbe/py_api/tests:$PYTHONPATH
```

Building budabackend
- `make -C bbe -j8 build_hw eager_backend`

Running a matmul test
- Grayskull: `python3 tests/test_tt_matmul.py -d gs -t matmul`
- Wormhole: `python3 tests/test_tt_matmul.py -d wh -t matmul`
