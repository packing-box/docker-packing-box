#!/usr/bin/env bats
load ./.init.sh

@test "run tool's help" {
  run_tool_help
}

