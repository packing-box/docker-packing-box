#!/usr/bin/env bats
load ./.init.sh

# TODO:
# ✗ setup
# ✗ test

@test "run tool's help" {
  run_tool_help
}

# ✓ clean
@test "create a temporary setup/test folder then clean it" {
  local TESTD1="/tmp/detector-setup-`openssl rand -hex 16`"
  local TESTD1="/tmp/packer-tests-`openssl rand -hex 16`"
  mkdir $TESTD1 $TESTD2
  run packing-box clean
  assert_file_not_exist $TESTD1
  assert_file_not_exist $TESTD2
}

# ✓ config
@test "change vt-api-key configuration setting" {
  packing-box config --vt-api-key "TEST_API_KEY"
  run packing-box list config
  assert_output --partial "TEST_API_KEY"
}

# ✓ list
@test "list configuration settings and check for the presence of some settings" {
  run packing-box list config
  for KEY in algorithms alterations analyzers detectors features packers references scenarios unpackers; do
    assert_output --partial "$KEY"
  done
}

# ✓ workspace
@test "view the current workspace" {
  packing-box workspace view
  run bats_pipe packing-box workspace view \| head -1
  assert_output --partial "$TEST_XP"
  run packing-box workspace view
  assert_output --partial "$TEST_DS"
}

