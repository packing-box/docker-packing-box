#!/usr/bin/env bats
load ./.init.sh

# TODO:
# ✗ browse
# ✗ export
# ✗ inspect
# ✗ select

@test "run tool's help" {
  run_tool_help
}

# ✓ show
@test "show statistics about the target alterations set" {
  run alteration show
  assert_output --partial 'Counts'
}

