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
    "test_tt_functional.py"
    "test_tt_functional.py"
    "test_tt_functional.py"
)
wh_tests=(
    "test_tt_tensor.py"
    "test_tt_matmul.py -t matmul"
    "test_tt_matmul.py -t layernorm"
    # multichip tests
    "test_tt_matmul.py -t matmul_2xchip"
    "test_tt_broadcast.py" 
)

# -------------------------------------------------------------------
# MAIN RUNNER
# -------------------------------------------------------------------
if [ "$device" == "gs" ]; then
    tests=("${gs_tests[@]}")
elif [ "$device" == "wh" ]; then
    tests=("${wh_tests[@]}")
else
    echo "Invalid device provided $device"
    echo "Usage: $0 <gs,wh>"
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
