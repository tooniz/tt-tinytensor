# TinyTensor

### WH Installation
directory structure looks something like this:
```bash
.
|-- budabackend
|-- tinytensor 
```

- from bbe, build backend `ARCH_NAME=wormhole make -j8 build_hw eager_backend loader/tests/test_eager_ops`
- change python path to be like this: `export PYTHONPATH=/localdev/cglagovich/budabackend/build/obj/py_api:/localdev/cglagovich/budabackend/py_api/tests:/localdev/cglagovich/tinytensor/:/localdev/cglagovich/tinytensor/src/:/localdev/cglagovich/tinytensor/tests/:$PYTHONPATH`
- install python requirements from `tinytensor/`: `pip install -r requirements.txt`
- from bbe repo, run `python ../tinytensor/tests/test_tt_matmul.py -d wh`
