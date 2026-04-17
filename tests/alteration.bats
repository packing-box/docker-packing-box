#!/usr/bin/env bats
load ./.init.sh

# NO TEST (interactive commands):
# ✗ browse
# ✗ inspect

@test "run tool's help" {
  run_tool_help
}

# ✓ show
@test "show statistics about the target alterations set" {
  run alteration show
  assert_output --partial 'Counts'
}

# ✓ select
@test "select and display alterations from the default alterations.yaml" {
  run alteration select --format PE -q 'keep == "True"'
  assert_output --partial 'description'
  assert_output --partial 'result'
}

# ✓ export
@test "export the default alterations.yaml to various formats" {
  for FMT in csv html json md tex txt xlsx xml yml; do
    run alteration export --output $FMT
    assert_file_exist "alterations-export.$FMT"
    rm "alterations-export.$FMT"
  done
}

