#!/usr/bin/env bats
load ./.init.sh

@test "run tool's help" {
  run_tool_help
}

@test "pack samples from $TEST_DS" {
  run dataset update $TEST_DS -f PE --source-dir ~/.wine64/drive_c/windows/system32 -n 10
  for PACKER in `list-working-packers`; do
    run packer $PACKER $TEST_DS
    assert_success
  done
}

