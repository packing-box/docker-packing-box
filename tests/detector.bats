#!/usr/bin/env bats
load ./.init.sh

@test "run tool's help" {
  run_tool_help
}

@test "detect samples from $TEST_DS" {
  dataset make $TEST_DS -f PE -p upx -n 10
  for DETECTOR in `list-working-detectors`; do
    run detector $TEST_DS -d $DETECTOR
    assert_success
  done
}

