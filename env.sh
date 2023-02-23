export PYTHONPATH=/localdev/cglagovich/budabackend/build/obj/py_api:/localdev/cglagovich/budabackend/py_api/tests:/localdev/cglagovich/tinytensor/:/localdev/cglagovich/tinytensor/src/:/localdev/cglagovich/tinytensor/tests/:$PYTHONPATH

source venv/bin/activate

export ARCH_NAME=wormhole
cd ../budabackend && make && make eager_backend && make loader/tests/test_eager_ops
