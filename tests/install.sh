#!/bin/bash
git submodule add -f https://github.com/bats-core/bats-core.git tests/.bats/bats
tests/bats/install.sh ~/.opt
git submodule add -f https://github.com/bats-core/bats-support.git tests/.bats/bats-support
git submodule add -f https://github.com/bats-core/bats-assert.git tests/.bats/bats-assert
git submodule add -f https://github.com/ztombol/bats-file.git tests/.bats/bats-file
