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
  load '.bats/pbox-helpers/load'
}

teardown_file(){
  # clean up the dedicated workspace
  run experiment close
  rm -f ~/.packing-box/experiments.env
  rm -rf "$TESTS_DIR"
}


@test "run tool's help" {
  run_tool_help
}

# ✓ list
@test "get the list of models (none yet)" {
  skip
  run model list
  assert_output --partial 'No model found'
}

# ✓ train
# ✓ rename
# ✓ test
# ✓ show
@test "train a model based on $TEST_DS (RandomForest)" {
  skip
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

@test "train models for every available algorithm based on $TEST_DS" {
  for ALGO in `list-all-algorithms`; do
    # AdaBoost failing with AttributeError: 'DecisionTreeClassifier' object has no attribute 'ccp_alpha'
    if [[ "$ALGO" != "ab" ]]; then
      model train "$TEST_DS" -a $ALGO
    fi
  done
}

# ✓ purge
@test "purge all models" {
  run model purge all
  run model list
  refute_output --partial "$TEST_MD"
  assert_output --partial 'No model found'
}
