#!/usr/bin/env bats
CREATE_EXP=0
load ./.init.sh

# NO TEST (interactive commands):
# ✗ edit

# TODO:
# ✗ compress
# ✗ play
# ✗ replay

# ✓ open
# ✓ close
# ✓ purge
@test "run tool's help" {
  run_tool_help
  experiment open "$TEST_XP"
  run_tool_help
  experiment close
  run experiment purge "$TEST_XP"
  refute_output --partial "$TEST_XP"
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
