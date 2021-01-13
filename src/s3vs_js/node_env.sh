#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

source $SCRIPT_DIR/../../scripts/env.sh $1
export s3vs_binary_dir=$S3DMM_BINARY_DIR

