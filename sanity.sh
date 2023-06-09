#!/usr/bin/env bash

export CI_RUNNER=1
PASS="\033[0;32m<PASSED>\033[0m\n"
FAIL="\033[0;31m<FAILED>\033[0m\n"
SCRIPT=$(readlink -f $0)
DIR=`dirname $SCRIPT`

device=${1:-NULL}
runner="python3"

# -------------------------------------------------------------------
# EDIT LISTS TO ADD NEW TESTS
# repeat the same random test if multiple loops are desired
# -------------------------------------------------------------------
gs_tests=(
    # "test_tt_tensor.py"
    "test_tt_matmul.py -t matmul"
    "test_tt_matmul.py -t layernorm"
    "test_tt_matmul.py -t matmul_gelu_matmul"
    "test_tt_functional.py"
)
wh_tests=(
    # "test_tt_tensor.py"
    "test_tt_matmul.py -t matmul_iterations"
    "test_tt_matmul.py -t layernorm"
    # nebula 2-chip tests
    "test_tt_broadcast.py" 
    "test_tt_matmul.py -t matmul_1d"
    "test_tt_matmul.py -t matmul_2xchip"
    "test_tt_matmul.py -t layernorm_2xchip"
    # galaxy multichip tests
    # "test_tt_matmul.py -t matmul_galaxy"
    # "test_tt_matmul.py -t matmul_gelu_matmul_galaxy"
)
whb0_tests=(
    "test_tt_matmul.py -t matmul"
    "test_tt_matmul.py -t layernorm"
    # nebula 2-chip tests
    "test_tt_matmul.py -t matmul_1d"
    "test_tt_matmul.py -t matmul_2xchip"
    "test_tt_matmul.py -t layernorm_2xchip"
)

# -------------------------------------------------------------------
# MAIN RUNNER
# -------------------------------------------------------------------
if [ "$device" == "gs" ] || [ "$device" == "wh" ] || [ "$device" == "whb0" ]; then
    suite=${device}_tests[@]
    tests=("${!suite}")
else
    echo "Invalid device provided $device"
    echo "Usage: $0 <gs,wh,whb0>"
    exit 1
fi
echo "Launching sanity tests on $device ..."

tid=0
result=0
status=$PASS
for test in "${tests[@]}"; do
    logfile=tt_test$tid.log
    printf "\033[1mRunning: $test \033[0m \t($logfile)\n"
    $runner $DIR/tests/$test -d $device 2>&1 >> $logfile
    RETCODE=$?
    if [ $RETCODE -eq 0 ]; then
      printf "  $PASS"
    else
      printf "  $FAIL"
      status=$FAIL
      result=1
    fi
    ((tid++))
done

printf "\nAll tests completed $status"
exit $result
