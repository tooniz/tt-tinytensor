#!/bin/sh

if [ ! -d "./bbe" ] 
then
    echo "Directory 'bbe' not found. Please create a symlink from your budabackend repo to 'bbe'" 
    exit 9999 # die with error code 9999
fi

export ARCH_NAME=wormhole
export ROOT=`git rev-parse --show-toplevel`
export BUDA_HOME=$ROOT/bbe
export PYTHONPATH=$ROOT:$ROOT/src:$ROOT/tests:$ROOT/bbe/build/obj/py_api:$ROOT/bbe/py_api/tests:$PYTHONPATH

echo "Building bbe..."
make -C bbe -j16 build_hw eager_backend
