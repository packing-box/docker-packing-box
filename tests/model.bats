#!/usr/bin/env bats
N=20
CONVERT=1
load ./.init.sh

# NO TEST (interactive commands):
# ✗ browse
# ✗ edit
# ✗ preprocess

# TODO:
# ✗ compare
# ✗ visualize

@test "run tool's help" {
  run_tool_help
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
  run dataset list
  assert_output --partial "$TEST_DS"
  run model train "$TEST_DS" -A rf
  MD_NAME="`list-models`"
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
      run model train "$TEST_DS" -A $ALGO
      assert_success
    fi
  done
}

# ✓ purge
@test "purge all models" {
  run model purge ALL
  run model list
  refute_output --partial "$TEST_MD"
  assert_output --partial 'No model found'
}

