#!/bin/bash
#
# NOTE: You may want to select a Qt version before running this script, e.g.,
# export QT_SELECT=qt5-11-2
#
SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

set -e

mkdir -p $SCRIPTDIR/builds/revise/release
cd $SCRIPTDIR/builds/revise/release

cmake -DCMAKE_BUILD_TYPE=Release "$@" $SCRIPTDIR
cpu_count=$(cat /proc/cpuinfo |grep processor| wc -l)
make -j$cpu_count

