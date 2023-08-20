run_tool_help() {
  local -r TOOL="`basename $BATS_TEST_FILENAME .bats`"
  local -r LOC="`which $TOOL`"
  local -r MSG="`grep __script__ $LOC | cut -d'"' -f2`"
  run $TOOL --help
  assert_output --partial $MSG
  assert_output --partial 'positional argument'
  assert_output --partial 'Usage examples'
  CMDS=$(grep _${TOOL}_completions ~/.bash_completion -c 1 -A 7 | grep compgen | cut -d'"' -f 2)
  if [[ ! $CMDS == \`*\` ]]; then
    for CMD in $CMDS; do
      run $TOOL $CMD --help
      assert_output --partial $MSG
      assert_output --partial 'extra arguments'
      assert_output --partial 'Usage example'  # NOT 'exampleS', as some commands have only one example !
    done
  fi
}
