#!/usr/bin/env bats

setup() {
  load '.bats/bats-support/load'
  load '.bats/bats-assert/load'
  load '.bats/bats-file/load'
  load '.bats/pbox-helpers/load'
}

@test "run tool's help" {
  run_tool_help
}

