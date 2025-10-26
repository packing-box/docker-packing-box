#!/usr/bin/env bats
load ./.init.sh

# TODO:
# ✗ alter
# ✗ disassemble
# ✗ fingerprint
# ✗ visualize

@test "run tool's help" {
  run_tool_help
}

# ✓ show
@test "show information about /bin/ls (built-in executable)" {
  run executable show /bin/ls
  assert_output --partial 'Hash'
  assert_output --partial 'Entry point'
  assert_output --partial 'Block entropy'
  refute_output --partial "Label"
}
@test "show information about $(basename -- $TEST_EXE) (from a dataset)" {
  run executable show $TEST_EXE
  assert_output --partial 'Label'
}

# ✓ features
@test "show features of $(basename -- $TEST_EXE) (from a dataset)" {
  run executable features $TEST_EXE
  assert_output --partial 'Features'
  assert_output --partial 'entropy'
}

