#!/bin/bash
cd "$( dirname "${BASH_SOURCE[0]}" )"

set -e
source ../../scripts/env.sh

source ./generators.sh

mkdir -p data
mkdir -p log/gen log/prep log/ren

gen_cube 1e2 1 4 1 2
gen_cube 1e3 1 5 1 2
gen_cube 1e7 1 8 1 2
gen_cube 1e7 10 8 1 2

gen_sphere 1e3 1 6 1 2
gen_sphere 1e4 1 7 1 2
gen_sphere 1e5 1 7 1 2
gen_sphere 1e5 10 7 1 2
gen_sphere 1e6 1 7 1 2
gen_sphere 1e7 1 8 1 2
#gen_sphere 1e8 1 8 1 2

gen_sphere 1e2 1 4 3 1 0
gen_sphere 1e3 1 5 3 1 0
gen_sphere 1e7 1 8 3 1 0

gen_circle 1e3 1 6 2 1 3
gen_circle 1e4 1 7 2 1 3
gen_circle 1e5 1 7 2 1 3
gen_circle 1e5 10 7 2 1 3
gen_circle 1e6 1 7 2 1 3
gen_circle 1e7 1 8 2 1 3
#gen_circle 1e8 1 8 2 1 4

gen_coleso mwave 256 10 8 0
gen_coleso mwave 512 1 8 1

# gen_coleso pointsource - - 7 0
# gen_coleso coaxial-1 - - 7 0
# gen_coleso coaxial-2 - - 7 0

