#
# pbox-helpers - Common Packing-Box dedicated assertions and helpers for Bats
#

#
# folder.bash
# -----------
#
# This module contains a function to check that a folder is not empty.
# This is required for asserting that files exist when a dataset has its files embedded in order to test that there was
#  no issue with processing the dataset.
#

# This test function first checks that the folder exists and then tests if this folder has files.
assert_folder_not_empty() {
  local -r folder="$1"
  if [[ ! -e "$folder" ]]; then
    local -r rem="$BATSLIB_FILE_PATH_REM"
    local -r add="$BATSLIB_FILE_PATH_ADD"
    batslib_print_kv_single 4 'path' "${folder/$rem/$add}" \
      | batslib_decorate 'folder does not exist' \
      | fail
  fi
  if [ -z "$(ls -A $folder)" ]; then
    local -r rem="$BATSLIB_FILE_PATH_REM"
    local -r add="$BATSLIB_FILE_PATH_ADD"
    batslib_print_kv_single 4 'path' "${folder/$rem/$add}" \
      | batslib_decorate 'folder is empty' \
      | fail
  fi
}
