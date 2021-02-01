#!/bin/bash
#
# Usage: source env.sh [debug|release]
#

# Do nothing if this script has already been invoked before
if [ -z "$REVISE_ENVIRONMENT_INITIALIZED" ]
then
    REVISE_SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
    export REVISE_ROOT_DIR=$(realpath $REVISE_SCRIPTS_DIR/..)
    REVISE_CUSTOM_SCRIPTS_DIR=$REVISE_SCRIPTS_DIR/custom

    [ "$1" == "debug" ] && REVISE_BUILD_TYPE=debug || REVISE_BUILD_TYPE=release

    if [ -f "$REVISE_CUSTOM_SCRIPTS_DIR/env.sh" ]; then
        echo found $REVISE_CUSTOM_SCRIPTS_DIR/env.sh, sourcing...
        source $REVISE_CUSTOM_SCRIPTS_DIR/env.sh
    fi

    [ -z "$REVISE_BUILD_DIR_RELEASE" ] && REVISE_BUILD_DIR_RELEASE=$REVISE_ROOT_DIR/builds/revise/release
    [ -z "$REVISE_BUILD_DIR_DEBUG" ] && REVISE_BUILD_DIR_DEBUG=$REVISE_ROOT_DIR/builds/revise/debug
    if [ -z "$REVISE_BUILD_DIR" ]; then
        [ "$REVISE_BUILD_TYPE" == "debug" ] && REVISE_BUILD_DIR=$REVISE_BUILD_DIR_DEBUG || REVISE_BUILD_DIR=$REVISE_BUILD_DIR_RELEASE
    fi
    [ -z "$REVISE_BINARY_DIR" ] && REVISE_BINARY_DIR=$REVISE_BUILD_DIR/bin

    if [ ! -d "$REVISE_BINARY_DIR" ]; then
        >&2 echo "ERROR: s3dmm binary directory does not exist: $REVISE_BINARY_DIR"
        return 1
    fi
    export REVISE_BINARY_DIR

    export PATH=$REVISE_BINARY_DIR:$REVISE_SCRIPTS_DIR:$PATH
    export LD_LIBRARY_PATH=$REVISE_ROOT_DIR/dist/lib:$REVISE_BINARY_DIR${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    export REVISE_ENVIRONMENT_INITIALIZED=true
fi
