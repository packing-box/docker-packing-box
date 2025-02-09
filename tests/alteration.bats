#!/usr/bin/env bats

# TODO:
# ✗ browse
# ✗ export
# ✗ inspect
# ✗ select


setup() {
  load '.bats/bats-support/load'
  load '.bats/bats-assert/load'
  load '.bats/bats-file/load'
  load '.bats/pbox-helpers/load'
}

@test "run tool's help" {
  run_tool_help
}

# ✓ show
@test "show statistics about the target alterations set" {
  run alteration show
  assert_output --partial 'Counts'
}

