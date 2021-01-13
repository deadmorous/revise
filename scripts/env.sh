#!/bin/bash
#
# Usage: source env.sh [debug|release]
#
S3DMM_SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
S3DMM_ROOT_DIR=$(realpath $S3DMM_SCRIPTS_DIR/..)
S3DMM_CUSTOM_SCRIPTS_DIR=$S3DMM_SCRIPTS_DIR/custom

[ "$1" == "debug" ] && S3DMM_BUILD_TYPE=debug || S3DMM_BUILD_TYPE=release

if [ -f "$S3DMM_CUSTOM_SCRIPTS_DIR/env.sh" ]; then
    echo found $S3DMM_CUSTOM_SCRIPTS_DIR/env.sh, sourcing...
    source $S3DMM_CUSTOM_SCRIPTS_DIR/env.sh
fi

[ -z "$S3DMM_BUILD_DIR_RELEASE" ] && S3DMM_BUILD_DIR_RELEASE=$S3DMM_ROOT_DIR/builds/revise/release
[ -z "$S3DMM_BUILD_DIR_DEBUG" ] && S3DMM_BUILD_DIR_DEBUG=$S3DMM_ROOT_DIR/builds/revise/debug
if [ -z "$S3DMM_BUILD_DIR" ]; then
    [ "$S3DMM_BUILD_TYPE" == "debug" ] && S3DMM_BUILD_DIR=$S3DMM_BUILD_DIR_DEBUG || S3DMM_BUILD_DIR=$S3DMM_BUILD_DIR_RELEASE
fi
[ -z "$S3DMM_BINARY_DIR" ] && S3DMM_BINARY_DIR=$S3DMM_BUILD_DIR/bin

if [ ! -d "$S3DMM_BINARY_DIR" ]; then
    >&2 echo "ERROR: s3dmm binary directory does not exist: $S3DMM_BINARY_DIR"
    return 1
fi

export PATH=$S3DMM_BINARY_DIR:$S3DMM_SCRIPTS_DIR:$PATH

export LD_LIBRARY_PATH=$S3DMM_ROOT_DIR/dist/lib:$S3DMM_BINARY_DIR${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

