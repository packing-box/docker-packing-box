#!/usr/bin/env bats

# TODO:
# ✗ alter
# ✗ disassemble
# ✗ fingerprint
# ✗ visualize


setup_file() {
  export TESTS_DIR="/tmp/tests-`openssl rand -hex 16`"
  export TEST_DS="DS01"
  export TEST_XP="XP01"
  # create a dedicated workspace for the tests
  echo -en "$TESTS_DIR" > ~/.packing-box/experiments.env
  run experiment open "$TEST_XP"
  # make a small dataset and take the name of its first sample
  run dataset make "$TEST_DS" -f PE -p upx -n 10
  export TEST_EXE="$TESTS_DIR/$TEST_XP/datasets/$TEST_DS/files/`ls $TESTS_DIR/$TEST_XP/datasets/$TEST_DS/files | head -1`"
}

setup() {
  load '.bats/bats-support/load'
  load '.bats/bats-assert/load'
  load '.bats/bats-file/load'
  load '.bats/pbox-helpers/load'
}

teardown_file(){
  # clean up the dedicated workspace
  run experiment close
  rm -f ~/.packing-box/experiments.env
  #rm -rf "$TESTS_DIR"
}


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

