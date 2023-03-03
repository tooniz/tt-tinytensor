#!/bin/sh

if [ ! -d "./bbe" ] 
then
    echo "Directory 'bbe' not found. Please create a symlink from your budabackend repo to 'bbe'" 
    exit 1
fi

if [ -z ${ARCH_NAME+x} ]
then
    echo "ARCH_NAME environment variable must be set! Choices: grayskull, wormhole, wormhole_b0" 
    exit 1
fi
export ROOT=`git rev-parse --show-toplevel`
export BUDA_HOME=$ROOT/bbe
export PYTHONPATH=$ROOT:$ROOT/src:$ROOT/tests:$ROOT/bbe/build/obj/py_api:$ROOT/bbe/py_api/tests:$PYTHONPATH

echo "Building bbe..."
make -C bbe -j16 build_hw eager_backend
