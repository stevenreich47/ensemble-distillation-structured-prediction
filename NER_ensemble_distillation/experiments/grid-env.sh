#! /usr/bin/env bash
set -euo pipefail


# This script sets up the grid environment for experiments in such a way that
# it should just work on any of our grids or in a non-grid environment.

# This script may be called directly, but should normally be sourced from
# experiment scripts.


# 'load_module' function attempts to load modules passed as arguments in order,
# and returns when the first one succeeds (or when they all fail)
load_module() {
  last_module=''
  for module in "${@:1}"; do
    if [[ $last_module ]]; then
      echo Unable to load module "$last_module", trying to fallback to $module
    fi
    last_module=$module
    if module load $module; then
      return 0
    fi
  done
  echo Failed to load module "$0" or any of its alternatives!
  return 1
}


# if the 'module' command exists, assume a grid environment and load modules
if command -v module; then
  load_module cuda10.1/toolkit cuda/10.1
  load_module cudnn/7.6.3_cuda10.1 cudnn/7.6.2
  load_module nccl/2.4.7_cuda10.1 nccl/2.4.7
  module load java/1.10.0 &> /dev/null || true # in some grid environments, this module doesn't exist
else
  echo running in non-grid environment
fi
