#!/bin/bash

set -e

if [ -z "$1" ]; then
    echo "clean.sh ERROR: '\$1' must contain directory name to be processed"
    exit
fi

LOG_DIR=$1

format2digits()
{
    [ ${#1} -eq 1 ] && echo 0$1 || echo $1
}

findLogs()
{
    local r=$1
    callback=$2
    ngpu=1
#    input_files=()
    while :
    do
        nwrk=$(( $ngpu * $r ))

        input_file=$LOG_DIR/VsRendererDrawTimestamps-$ngpu-$nwrk-*.log
        if [ -f $input_file ]; then
            $callback $r $ngpu $nwrk $input_file
#            input_files+=("$input_file")
        fi

        ngpu=$(( $ngpu * 2 ))
        if [ $ngpu -gt 100 ]; then
            break
        fi
    done
#    echo $input_files
}

filter_file() {
    tsv-filter -H --not-empty level "$3" \
    | keep-header -- sed "s/^/$1\t$2\t/" \
    | sed "s/^level/workers per GPU\tGPUs\tlevel/" \
    >"$4"
}

processLog()
{
    local r=$1
    local ngpu=$2
    local nwrk=$3
    local input_file=$4

    echo Processing $input_file
    vis_timestamps --input $input_file --mode task_timeline
    vis_timestamps --input $input_file --mode task_stage_timeline
    vis_timestamps --input $input_file --mode task_duration --favg 1-$ --proj "level|duration|^.*_count$|^.*_avg$" >$input_file.dir/task_avg_duration.tsv
    vis_timestamps --input $input_file --mode task_stage_duration --favg 1-$ --proj "level|duration|^.*_count$|^.*_avg$" >$input_file.dir/task_stage_avg_duration.tsv

    filter_file $r $ngpu $input_file.dir/task_avg_duration.tsv tmp/${r}_${ngpu}_total_avg.tsv
    filter_file $r $ngpu $input_file.dir/task_stage_avg_duration.tsv tmp/${r}_${ngpu}_stage_avg.tsv
}

mergeLogs()
{
    basename="$1"
    tsv-append -H tmp/*_$basename \
    | keep-header -- sort -s -k 3,3 \
    | keep-header -- sort -s -k 1,1 \
    | keep-header -- sort -s -k 2,2 \
    >$LOG_DIR/$basename
}

mkdir -p tmp
for r in 1 2 4; do
    echo workers per GPU: $r
    findLogs $r processLog
    mergeLogs total_avg.tsv
    mergeLogs stage_avg.tsv
done
rm -rf tmp
