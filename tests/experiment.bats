#!/usr/bin/env bats

# NO TEST (interactive commands):
# ✗ edit

# TODO:
# ✗ compress


setup_file() {
  export TESTS_DIR="/tmp/tests-`openssl rand -hex 16`"
  export TEST_DS="DS01"
  export TEST_MD="MD01"
  export TEST_XP="XP01"
  # create a dedicated workspace for the tests
  echo -en "$TESTS_DIR" > ~/.packing-box/experiments.env
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


# ✓ open
# ✓ close
# ✓ purge
@test "run tool's help" {
  run_tool_help
  experiment open temp
  run_tool_help
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

# ✓ commit
@test "create $TEST_DS in $TEST_XP and commit command" {
  dataset make test-upx -n 10 -f PE -p upx
  echo -e "dataset -v make test-upx -n 10 -f PE -p upx" >> ~/.bash_history
  run experiment commit -f
  local -r CMDRC="$TESTS_DIR/$TEST_XP/commands.rc"
  assert_file_exist $CMDRC
  run cat $CMDRC
  assert_output --partial 'dataset -v make test-upx -n 10 -f PE -p upx'
}
