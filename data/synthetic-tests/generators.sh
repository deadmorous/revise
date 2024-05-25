#!/bin/bash
GENERATORS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

gen_general () {
    local label=$1
    local type=$2
    local dim=$3
    local N=$4
    local tn=$5
    local depth=$6
    local br=$7
    local max_level=$8
    local basename=${label}_N${N}_t${tn}_D${depth}_br${br}_L${max_level}
    local name=$basename.bin
    local ncpu=10
    
    if [ ! -d data/$basename ]
    then
        if [ "$tn" == "1" ]
        then
            mkdir -p data/$basename
            name=$basename/$label.bin
        fi

        echo "# label=$label, type=$type, dim=$dim, N=$N, tn=$tn, depth=$depth, br=$br, max_level=$max_level
# basename=$basename

# GENERATE
tecplot_gen --type $type --dim $dim --N $N --tn $tn --output data/$name

# PREPROCESS
s3dmm_prep --dim $dim --depth $depth --brefine $br --mesh_file data/$name --max_full_level $max_level --max_level $max_level -t $ncpu

" >> commands.log

        { time tecplot_gen --type $type --dim $dim --N $N --tn $tn --output data/$name; } |& tee log/gen/$basename.log
        s3dmm_prep --dim $dim --depth $depth --brefine $br --mesh_file data/$name --max_full_level $max_level --max_level $max_level -t $ncpu |& tee log/prep/$basename.log
    fi
}

gen_coleso () {
    local type=$1
    local cells=$2
    local tn=$3
    local depth=$4
    local max_level=$5
    local basename=${type}_c${cells}_t${tn}_D${depth}_L${max_level}
    local output_dir=data/$basename
    local ncpu=10
    if [ ! -d $output_dir ]
    then
        echo "# type=$type, cells=$cells, tn=$tn, depth=$depth, max_level=$max_level
# output_dir=$output_dir

# GENERATE &  PREPROCESS
s3dmm_prep --mode coleso --exact_config $GENERATORS_DIR/config/$type.json --exact_cells $cells --exact_time_steps $tn --depth $depth --max_full_level $max_level --output_dir $output_dir -t $ncpu


" >> commands.log

        s3dmm_prep --mode coleso --exact_config $GENERATORS_DIR/config/$type.json --exact_cells $cells --exact_time_steps $tn --depth $depth --max_full_level $max_level --output_dir $output_dir -t $ncpu |& tee log/prep/$basename.log
    fi
}

gen_cube () {
    gen_general cube cube 3 "$@"
}

gen_square () {
    gen_general square cube 2 "$@"
}

gen_sphere () {
    gen_general sphere sphere 3 "$@"
}

gen_circle () {
    gen_general circle sphere 2 "$@"
}

