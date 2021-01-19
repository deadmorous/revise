#!/bin/bash
REVISE_SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

$REVISE_SCRIPTS_DIR/../src/webserver/start.sh "$@"
