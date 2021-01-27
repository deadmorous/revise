#!/bin/bash

# $1 --- the directory with .log files to process. Must contain files VsRendererDrawTimestamps-*.log
# All output files will be written also here.
REVISE_SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
REVISE_ROOT_DIR=$(realpath $REVISE_SCRIPTS_DIR/..)

LOG_DIR=$1

TSV_UTILS_BIN=$REVISE_ROOT_DIR/third_parties/tsv-utils/bin
REVISE_BIN=$REVISE_SCRIPTS_DIR/builds/revise/release/bin
PATH=$PATH:$TSV_UTILS_BIN:$REVISE_BIN

set -e

# Generate tables and images
for log in $LOG_DIR/VsRendererDrawTimestamps-*.log; do
    echo Processing $log
    vis_timestamps --input $log --mode task_timeline
    vis_timestamps --input $log --mode task_stage_timeline
    vis_timestamps --input $log --mode task_duration --favg 1-$ --proj "level|duration|^.*_count$|^.*_avg$" >$log.dir/task_avg_duration.tsv
    vis_timestamps --input $log --mode task_stage_duration --favg 1-$ --proj "level|duration|^.*_count$|^.*_avg$" >$log.dir/task_stage_avg_duration.tsv
done

# Process tables (collect data from different tables into other tables)
# Note: this requires tsv-utils

for basename in task_avg_duration task_stage_avg_duration; do
    mkdir -p tmp

    input_files=""
    for n in 1 2 4 8; do
        input_file=$LOG_DIR/VsRendererDrawTimestamps-0$n-0$n-16.log.dir/${basename}.tsv
	if [ -f "$input_file" ]; then
        	tmp_file=tmp/$n.tsv
        	tsv-filter -H --not-empty level $input_file >$tmp_file
        	input_files=${input_files:+$input_files }$tmp_file
	fi
    done

    if [ ! -z "$input_files" ]; then
	tsv-append -H -t $input_files |keep-header -- sort -s -k 2,2 >tmp/avg_durations_1w.tsv
    fi

    input_files=""
    for n in 01-02 02-04 04-08 08-16; do
        input_file=$LOG_DIR/VsRendererDrawTimestamps-$n-16.log.dir/${basename}.tsv
	if [ -f "$input_file" ]; then
        	tmp_file=tmp/$n.tsv
        	tsv-filter -H --not-empty level $input_file >$tmp_file
        	input_files=${input_files:+$input_files }$tmp_file
	fi
    done

    if [ ! -z "$input_files" ]; then
    	tsv-append -H -t $input_files |keep-header -- sort -s -k 2,2 >tmp/avg_durations_2w.tsv
    fi

    input_files=""
    for n in 01-04 02-08 04-16 08-32; do
        input_file=$LOG_DIR/VsRendererDrawTimestamps-$n-16.log.dir/${basename}.tsv
	if [ -f "$input_file" ]; then
        	tmp_file=tmp/$n.tsv
        	tsv-filter -H --not-empty level $input_file >$tmp_file
        	input_files=${input_files:+$input_files }$tmp_file
	fi
    done
    if [ ! -z "$input_files" ]; then
    	tsv-append -H -t $input_files |keep-header -- sort -s -k 2,2 >tmp/avg_durations_4w.tsv
    fi

    if [ -f tmp/avg_durations_1w.tsv -a -f tmp/avg_durations_2w.tsv -a -f tmp/avg_durations_4w.tsv ]; then
	    tsv-append -H tmp/avg_durations_1w.tsv tmp/avg_durations_2w.tsv tmp/avg_durations_4w.tsv |keep-header -- sort -s -k 2,2 >$LOG_DIR/${basename}_all.tsv
    fi

    rm -rf tmp/
done
