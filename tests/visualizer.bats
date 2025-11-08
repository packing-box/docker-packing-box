#!/usr/bin/env bats
load ./.init.sh

# TODO:
# ✗ compare
# ✗ features
# ✗ find
# ✗ plot
# ✗ remove

@test "run tool's help" {
  run_tool_help
}

@test "dummy test" {
  run ls -lah "$SAMPLES_DIR"
  assert_output --partial "calc"
}

