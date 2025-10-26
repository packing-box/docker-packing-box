#!/usr/bin/env bats
LOAD_ONLY=1
load ./.init.sh

@test "run tool's help" {
  run_tool_help
}

