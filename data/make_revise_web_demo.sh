#!/bin/bash
cd "$( dirname "${BASH_SOURCE[0]}" )"

set -e

./synthetic-tests/make_data.sh

source ../scripts/env.sh

revise_webdata.js install demo.json
revise_webdata.js clear
revise_webdata.js add synthetic-tests/data/sphere_N1e4_t1_D7_br1_L2/sphere.bin "Sphere 10K D7 L2"
revise_webdata.js add synthetic-tests/data/sphere_N1e5_t10_D7_br1_L2.bin "Sphere 100K x 10 D7 L2"
revise_webdata.js add synthetic-tests/data/sphere_N1e7_t1_D8_br1_L2/sphere.bin "Sphere 10M D8 L2"
revise_webdata.js add synthetic-tests/data/sphere_N1e7_t1_D8_br3_L1/sphere.bin "Sphere 10M D8 L1 br3"
revise_webdata.js add synthetic-tests/data/cube_N1e3_t1_D5_br1_L2/cube.bin "Cube 1K D5 L2"
revise_webdata.js add synthetic-tests/data/cube_N1e7_t10_D8_br1_L2.bin "Cube 10M x 10 D8 L2"
revise_webdata.js add synthetic-tests/data/mwave_c256_t10_D8_L0.coleso "MWave 17M x 10 D8 L0"
revise_webdata.js add synthetic-tests/data/mwave_c512_t1_D8_L1/data.coleso "MWave 135M D8 L1"

