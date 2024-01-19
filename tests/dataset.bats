#!/usr/bin/env bats

# NO TEST (interactive commands):
# ✗ browse
# ✗ edit
# ✗ preprocess

# TODO:
# ✗ export
# ✗ ingest
# ✗ update


setup_file() {
  export TESTS_DIR="/tmp/tests-`openssl rand -hex 16`"
  export TEST_DS1="DS01"
  export TEST_DS2="DS02"
  export TEST_DS3="DS03"
  export TEST_XP="XP01"
  # ensure that we are not in the scope of an opened experiment
  run experiment close
  # create a dedicated workspace for the tests
  echo -en "$TESTS_DIR" > ~/.packing-box/experiments.env
  run experiment open "$TEST_XP"
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

# ✓ list
@test "get the list of datasets (none yet)" {
  run dataset list
  assert_output --partial 'No dataset found'
}

# ✓ fix
# ✓ make
# ✓ show
# ✓ view
@test "make $TEST_DS1 (10 PE samples whose some are packed with UPX)" {
  run dataset make "$TEST_DS1" -n 10 -f PE -p upx
  assert_folder_not_empty "$TESTS_DIR/$TEST_XP/datasets/$TEST_DS1/files"
  assert_output --partial 'Source directories'
  assert_output --partial 'Used packers'
  assert_output --partial '#Executables'
  run dataset fix "$TEST_DS1"
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
# state:
#  - DS01: with files

# ✓ convert
@test "convert $TEST_DS1 to a fileless dataset" {
  run dataset convert "$TEST_DS1"
  assert_output --partial 'Converting to fileless dataset'
  assert_output --partial 'Size of new dataset'
  assert_file_not_exist "$TESTS_DIR/$TEST_XP/datasets/$TEST_DS1/files"
  assert_file_exist "$TESTS_DIR/$TEST_XP/datasets/$TEST_DS1/features.json"
}
# state:
#  - DS01: fileless

# ✓ rename
@test "rename $TEST_DS1 to $TEST_DS2" {
  run dataset rename "$TEST_DS1" "$TEST_DS2"
  assert_file_not_exist "$TESTS_DIR/$TEST_XP/datasets/$TEST_DS2/files"
  run dataset list
  refute_output --partial "$TEST_DS1"
  assert_output --partial "$TEST_DS2"
}
# state:
#  - DS02: fileless

# ✓ merge
@test "merge a new $TEST_DS1 with $TEST_DS2 into $TEST_DS3" {
  run dataset make "$TEST_DS1" -n 10 -f PE -p upx
  assert_folder_not_empty "$TESTS_DIR/$TEST_XP/datasets/$TEST_DS1/files"
  run dataset merge "$TEST_DS1" "$TEST_DS2" -n "$TEST_DS3"
  assert_file_not_exist "$TESTS_DIR/$TEST_XP/datasets/$TEST_DS3/files"
  run dataset list
  assert_output --partial "$TEST_DS1"
  assert_output --partial "$TEST_DS2"
  assert_output --partial "$TEST_DS3"
}
# state:
#  - DS01: with files
#  - DS02: fileless
#  - DS03: fileless

# ✓ purge
@test "purge $TEST_DS1" {
  run dataset purge "$TEST_DS1"
  run dataset list
  refute_output --partial "$TEST_DS1"
  assert_output --partial "$TEST_DS2"
  assert_output --partial "$TEST_DS3"
}
# state:
#  - DS02: fileless
#  - DS03: fileless

# ✓ select
# ✓ remove
# ✓ revert
@test "select to $TEST_DS1, remove UPX-samples from $TEST_DS2 then revert it" {
  run dataset select "$TEST_DS2" "$TEST_DS1" --query "label == 'upx'"
  run dataset show "$TEST_DS1"
  refute_output --partial '#Executables: 10'
  run dataset show "$TEST_DS2"
  assert_output --partial '#Executables: 10'
  run dataset remove "$TEST_DS2" --query "label == 'upx'"
  run dataset show "$TEST_DS2"
  refute_output --partial '#Executables: 10'
  run dataset revert "$TEST_DS2"
  run dataset show "$TEST_DS2"
  assert_output --partial '#Executables: 10'
}
# state:
#  - DS01: fileless
#  - DS02: fileless
#  - DS03: fileless

# ✓ plot
@test "plot distributions of $TEST_DS2" {
  LABELS="$TESTS_DIR/$TEST_XP/figures/$TEST_DS2/labels.png"
  assert_file_not_exist "${LABELS}"
  run dataset plot labels "$TEST_DS2"
  assert_file_exist "${LABELS}"
  run dataset purge "$TEST_DS1"
  run dataset make "$TEST_DS1" -n 10 -f PE -p upx
  assert_folder_not_empty "$TESTS_DIR/$TEST_XP/datasets/$TEST_DS1/files"
  ENTROPY="$TESTS_DIR/$TEST_XP/figures/$TEST_DS1/features/entropy.png"
  assert_file_not_exist "${ENTROPY}"
  run dataset plot features "$TEST_DS1" entropy
  assert_file_exist "${ENTROPY}"
}
# state:
#  - DS01: with files
#  - DS02: fileless
#  - DS03: fileless

# ✗ alter
@test "alter $TEST_DS1" {
  skip  #TODO
  run dataset alter 
}
