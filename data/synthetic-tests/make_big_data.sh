#!/bin/bash
cd "$( dirname "${BASH_SOURCE[0]}" )"

set -e
source ../../scripts/env.sh

source ./generators.sh

mkdir -p data
mkdir -p log/gen log/prep log/ren

gen_coleso mwave 1024 1 9 1

