#!/bin/bash

if [ -z "$1" ]; then
    echo "clean.sh ERROR: '\$1' must contain directory name to be cleaned"
    exit
fi
rm -rf $1/*.log.dir

