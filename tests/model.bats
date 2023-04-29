#!/usr/bin/env bats

# NO TEST (interactive commands):
# ✗ browse
# ✗ edit
# ✗ preprocess

# TODO:
# ✗ compare
# ✗ visualize


setup_file() {
  export TESTS_DIR="/tmp/tests-`openssl rand -hex 16`"
  export TEST_DS="DS01"
  export TEST_MD="MD01"
  export TEST_XP="XP01"
  # create a dedicated workspace for the tests
  echo -en "$TESTS_DIR" > ~/.packing-box/experiments.env
  run experiment open "$TEST_XP"
  # prepare a dataset
  run dataset make "$TEST_DS" -n 20 -f PE -p upx
  run dataset convert "$TEST_DS"
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


@test "run tool's help" {
  run model --help
  assert_output --partial 'Model'
  assert_output --partial 'positional arguments'
  assert_output --partial 'Usage examples'
  for CMD in "browse compare edit list preprocess purge rename show test train visualize"; do
    run dataset $CMD --help
  done
}

# ✓ list
@test "get the list of models (none yet)" {
  run model list
  assert_output --partial 'No model found'
}

# ✓ train
# ✓ rename
# ✓ test
# ✓ show
@test "train a model based on $TEST_DS (RandomForest)" {
  run model train "$TEST_DS" -a rf
  MD_NAME="`mdlst`"
  assert_output --partial "Name: $MD_NAME"
  assert_output --partial 'Classification metrics'
  assert_output --partial 'Parameters'
  run model rename "$MD_NAME" "$TEST_MD"
  run model list
  assert_output --partial "$TEST_MD"
  refute_output --partial "$MD_NAME"
  run model test "$TEST_MD" "$TEST_DS"
  assert_output --partial "Test results for: $TEST_DS"
  assert_output --partial 'Classification metrics'
  run model show "$TEST_MD"
  assert_output --partial 'Model characteristics'
  assert_output --partial 'Reference dataset'
  assert_output --partial "$TEST_MD"
  assert_output --partial "$TEST_DS"
}

# ✓ purge
@test "purge $TEST_MD" {
  run model purge all
  run model list
  refute_output --partial "$TEST_MD"
  assert_output --partial 'No model found'
}
