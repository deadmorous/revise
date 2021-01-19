#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

source $SCRIPT_DIR/../../scripts/env.sh $1

if [ -f "$REVISE_CUSTOM_SCRIPTS_DIR/webserver_env.sh" ]; then
    echo found $REVISE_CUSTOM_SCRIPTS_DIR/webserver_env.sh, sourcing...
    source $REVISE_CUSTOM_SCRIPTS_DIR/webserver_env.sh
fi

[ -z "$s3vs_binary_dir" ] && s3vs_binary_dir=$REVISE_BINARY_DIR
[ -z "$s3vs_problem_list_file" ] && s3vs_problem_list_file=$SCRIPT_DIR/configs/problems.json
[ -z "$s3vs_sendframe_port" ] && s3vs_sendframe_port=1234
[ -z "$kill_vsc_interval" ] && kill_vsc_interval=60000
[ -z "$requests_log_path" ] && requests_log_path=$SCRIPT_DIR/logs/requests.log
[ -z "$timedata_log_path" ] && timedata_log_path=$SCRIPT_DIR/logs/timedata.log

export s3vs_binary_dir
export s3vs_problem_list_file
export s3vs_sendframe_port
export kill_vsc_interval
export requests_log_path
export timedata_log_path

