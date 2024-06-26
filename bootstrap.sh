#!/bin/bash
#
# NOTE: You may want to select a Qt version before running this script, e.g.,
# export QT_SELECT=qt6
#
SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

set -e

# Parse command line
POSITIONAL=()
while [[ $# -gt 0 ]]
do
    echo processing argument "$1"
    case "$1" in
        --with-tsv-utils)
            WITH_TSV_UTILS=YES
            shift
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

# Update third party modules
git submodule update --init --recursive

mkdir -p $SCRIPTDIR/builds
mkdir -p $SCRIPTDIR/dist

# Prepare to build and install third parties
export CMAKE_PREFIX_PATH=$(qmake -query QT_INSTALL_PREFIX)
cd $SCRIPTDIR/builds
cpu_count=$(cat /proc/cpuinfo |grep processor| wc -l)

# Build and install VL
mkdir -p vl/debug && cd vl/debug
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=$SCRIPTDIR/dist -DVL_GUI_QT6_SUPPORT=ON $SCRIPTDIR/third_parties/VisualizationLibrary
make -j$cpu_count install
cd ../..
mkdir -p vl/release && cd vl/release
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$SCRIPTDIR/dist -DVL_GUI_QT6_SUPPORT=ON $SCRIPTDIR/third_parties/VisualizationLibrary
make -j$cpu_count install
cd ../..

# Build and install silver_bullets
mkdir -p silver_bullets/release && cd silver_bullets/release
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$SCRIPTDIR/dist $SCRIPTDIR/third_parties/silver_bullets
make -j$cpu_count install
cd ../..

cd ..

if [ "$WITH_TSV_UTILS" = "YES" ]; then
    # Build tsv-utils
    # Note: you will need the D language! To install on Ubuntu, run
    # sudo snap install --classic dmd
    echo "Building tsv-utils"
    cd third_parties/tsv-utils
    make -j$cpu_count
    cd -
fi
