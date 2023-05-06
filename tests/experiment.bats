#!/usr/bin/env bats

# NO TEST (interactive commands):
# ✗ edit

# TODO:
# ✗ commit
# ✗ compress


setup_file() {
  export TESTS_DIR="/tmp/tests-`openssl rand -hex 16`"
  export TEST_DS="DS01"
  export TEST_XP="XP01"
  # create a dedicated workspace for the tests
  echo -en "$TESTS_DIR" > ~/.packing-box/experiments.env
}

setup() {
  load '.bats/bats-support/load'
  load '.bats/bats-assert/load'
  load '.bats/bats-file/load'
}

teardown_file(){
  # clean up the dedicated workspace
  run experiment close
  rm -f ~/.packing-box/experiments.env
  rm -rf "$TESTS_DIR"
}


# ✓ open
# ✓ close
# ✓ purge
@test "run tool's help" {
  run experiment --help
  assert_output --partial 'Experiment'
  assert_output --partial 'positional argument'
  assert_output --partial 'Usage examples:'
  for CMD in compress list open purge; do
    experiment $CMD --help
  done
  experiment open temp
  for CMD in close commit compress edit list show; do
    experiment $CMD --help
  done
  experiment close
  run experiment purge temp
  refute_output --partial 'temp'
}

# ✓ list
@test "get the list of experiments (none yet)" {
  run experiment list
  assert_output --partial 'No experiment found'
}

# ✓ show
@test "create $TEST_XP and show its content" {
  run experiment close
  experiment open "$TEST_XP"
  assert_file_exist "$TESTS_DIR/$TEST_XP/conf"
  assert_file_exist "$TESTS_DIR/$TEST_XP/datasets"
  assert_file_exist "$TESTS_DIR/$TEST_XP/models"
  run experiment show
  assert_output --partial 'No dataset found'
  assert_output --partial 'No model found'
}
