#!/usr/bin/env bats

setup_file() {
  if ! [[ "${LOAD_ONLY:=0}" -ne 0 ]]; then
    export EXPS_DIR=$(cat ~/.packing-box/experiments.env 2>/dev/null || echo "")
    export EXP_DIR=$(cat ~/.packing-box/experiment.env 2>/dev/null || echo "")
    export TESTS_DIR="/tmp/tests-`openssl rand -hex 16`"
    export TEST_DS="DS01"
    export TEST_DS1="DS01"
    export TEST_DS2="DS02"
    export TEST_DS3="DS03"
    export TEST_MD="MD01"
    export TEST_XP="XP01"
    # if an experiment was currently open, close it
    if [[ -n "$EXP_DIR" ]]; then
      run experiment close
    fi
    # create a dedicated workspace for the tests
    echo -en "$TESTS_DIR" > ~/.packing-box/experiments.env
    if [[ "${CREATE_EXP:=1}" -ne 0 ]]; then run experiment open "$TEST_XP"; fi
    # make a small dataset and take the name of its first sample
    if [[ "${CREATE_DS:=1}" -ne 0 ]]; then
      run dataset make "$TEST_DS" -f PE -p upx -n ${N:=10}
      if [[ "${CONVERT:=0}" -ne 0 ]]; then run dataset convert "$TEST_DS"; fi
    fi
    export TEST_EXE="$TESTS_DIR/$TEST_XP/datasets/$TEST_DS/files/`ls $TESTS_DIR/$TEST_XP/datasets/$TEST_DS/files | head -1`"
  fi
}

setup() {
  load '.bats/bats-support/load'
  load '.bats/bats-assert/load'
  load '.bats/bats-file/load'
  load '.bats/pbox-helpers/load'
}

teardown_file(){
  if ! [[ "${LOAD_ONLY:=0}" -ne 0 ]]; then
    # clean up the dedicated workspace
    run experiment close
    rm -f ~/.packing-box/experiments.env
    rm -rf "$TESTS_DIR"
    # restore the previous experiments workspace if not default
    if [[ -n "$EXPS_DIR" ]]; then
      echo -en "$EXPS_DIR" > ~/.packing-box/experiments.env
    fi
    # restore previously open experiment if any
    if [[ -n "$EXP_DIR" ]]; then
      experiment open `basename "$EXP_DIR"`
    fi
  fi
}

