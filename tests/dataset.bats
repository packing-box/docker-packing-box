#!/usr/bin/env bats

# NO TEST (interactive commands):
# ✗ browse
# ✗ edit
# ✗ preprocess

# TODO:
# ✗ alter
# ✗ export
# ✗ fix
# ✗ ingest
# ✗ revert
# ✗ update


setup_file() {
  export TESTS_DIR="/tmp/tests-`openssl rand -hex 16`"
  export TEST_DS1="DS01"
  export TEST_DS2="DS02"
  export TEST_DS3="DS03"
  export TEST_XP="XP01"
  # create a dedicated workspace for the tests
  echo -en "$TESTS_DIR" > ~/.packing-box/experiments.env
  run experiment open "$TEST_XP"
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
  run dataset --help
  assert_output --partial 'Dataset'
  assert_output --partial 'positional arguments'
  assert_output --partial 'Usage examples'
  for CMD in "alter browse convert edit export fix ingest list make merge plot preprocess purge remove rename revert select show update view"; do
    run dataset $CMD --help
  done
}

# ✓ list
@test "get the list of datasets (none yet)" {
  run dataset list
  assert_output --partial 'No dataset found'
}

# ✓ make
# ✓ show
# ✓ view
@test "make $TEST_DS1 (10 PE samples whose some are packed with UPX)" {
  run dataset make "$TEST_DS1" -n 10 -f PE -p upx
  assert_output --partial 'Source directories'
  assert_output --partial 'Used packers'
  assert_output --partial '#Executables'
  run dataset list
  assert_output --partial 'Datasets'
  assert_output --partial "$TEST_DS1"
  run dataset show "$TEST_DS1"
  assert_output --partial 'Dataset characteristics'
  assert_output --partial 'Executables per size'
  assert_output --partial "Executables' metadata and labels"
  run dataset view "$TEST_DS1"
  assert_output --partial 'Sources'
  assert_output --partial 'Path'
  assert_output --partial 'Label'
}

# ✓ convert
@test "convert $TEST_DS1 to a fileless dataset" {
  run dataset convert "$TEST_DS1"
  assert_output --partial 'Converting to fileless dataset'
  assert_output --partial 'Size of new dataset'
}

# ✓ rename
@test "rename $TEST_DS1 to $TEST_DS2" {
  run dataset rename "$TEST_DS1" "$TEST_DS2"
  run dataset list
  refute_output --partial "$TEST_DS1"
  assert_output --partial "$TEST_DS2"
}

# ✓ merge
@test "merge a new $TEST_DS1 with $TEST_DS2 into $TEST_DS3" {
  run dataset make "$TEST_DS1" -n 10 -f PE -p upx
  run dataset merge "$TEST_DS1" "$TEST_DS2" -n "$TEST_DS3"
  run dataset list
  assert_output --partial "$TEST_DS1"
  assert_output --partial "$TEST_DS2"
  assert_output --partial "$TEST_DS3"
}

# ✓ purge
@test "purge $TEST_DS1" {
  run dataset purge "$TEST_DS1"
  run dataset list
  refute_output --partial "$TEST_DS1"
  assert_output --partial "$TEST_DS2"
  assert_output --partial "$TEST_DS3"
}

# ✓ select
# ✓ remove
@test "select to $TEST_DS1 and remove UPX-samples from $TEST_DS2" {
  run dataset select "$TEST_DS2" "$TEST_DS1" --query "label == 'upx'"
  run dataset show "$TEST_DS1"
  refute_output --partial '#Executables: 10'
  run dataset show "$TEST_DS2"
  assert_output --partial '#Executables: 10'
  run dataset remove "$TEST_DS2" --query "label == 'upx'"
  refute_output --partial '#Executables: 10'
}

# ✓ plot
@test "plot distributions of $TEST_DS2" {
  FILE="$TESTS_DIR/$TEST_XP/figures/$TEST_DS2"
  assert_file_not_exist "${FILE}_labels.png"
  run dataset plot labels "$TEST_DS2"
  assert_file_exist "${FILE}_labels.png"
  assert_file_not_exist "${FILE}_features_entropy.png"
  run dataset plot features "$TEST_DS2" entropy
  assert_file_exist "${FILE}_features_entropy.png"
}
