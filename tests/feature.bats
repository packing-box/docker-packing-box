#!/usr/bin/env bats


setup() {
  load '.bats/bats-support/load'
  load '.bats/bats-assert/load'
  load '.bats/bats-file/load'
  load '.bats/pbox-helpers/load'
}

@test "run tool's help" {
  run_tool_help
}

# ✓ show
@test "show statistics about the target features set" {
  run feature show
  assert_output --partial 'Counts per category'
  assert_output --partial 'Counts per processing time'
  assert_output --partial 'Counts per time complexity'
}

# ✓ select
@test "select and display features from the default features.yaml" {
  run feature select --format PE -q 'keep == "True"'
  assert_output --partial 'description'
  assert_output --partial 'result'
}

# ✓ export
@test "export the default features.yaml to various formats" {
  for FMT in csv html json md tex txt xlsx xml yml; do
    run feature export --output $FMT
    assert_file_exist "features-export.$FMT"
    rm "features-export.$FMT"
  done
}

# ✓ compute
@test "show features of /bin/ls" {
  run feature compute /bin/ls
  assert_output --partial 'Features'
  assert_output --partial 'entropy'
}

