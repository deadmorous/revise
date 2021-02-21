#!/bin/bash

set -e

[ -z "$1" ] && REVISE_DATASET_DEFAULT=cascade || REVISE_DATASET="$1"

# Set up environment, if necessary
source $( dirname "${BASH_SOURCE[0]}" )/env.sh

# Guess number of GPUs installed on the system
if [ ! -z $(which nvidia-smi) ]
then
    GPU_NAME_LIST=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader |sed "s/ /_/g")
    GUESSED_GPU_COUNT=$(echo $GPU_NAME_LIST |wc -w)
    echo "GPUs found ($GUESSED_GPU_COUNT)":
    echo $GPU_NAME_LIST
    # GUESSED_MACHINE_NAME=$(hostname)__$(echo $GPU_NAME_LIST |cut -d " " -f 1)
    GUESSED_MACHINE_NAME=$(hostname)
else
    GUESSED_GPU_COUNT=1
    GUESSED_MACHINE_NAME=dark_horse
fi

# Read the total number of GPUs
read -p "Total number of GPUs [$GUESSED_GPU_COUNT]: " -i 1 REVISE_TOTAL_GPUS
REVISE_TOTAL_GPUS=${REVISE_TOTAL_GPUS:-$GUESSED_GPU_COUNT}
echo REVISE_TOTAL_GPUS=$REVISE_TOTAL_GPUS

# GUESSED_MACHINE_NAME=${GUESSED_MACHINE_NAME}_x${REVISE_TOTAL_GPUS}

read -p "Machine name [$GUESSED_MACHINE_NAME]: " -i 1 REVISE_MACHINE
REVISE_MACHINE=${REVISE_MACHINE:-$GUESSED_MACHINE_NAME}
echo REVISE_MACHINE=$REVISE_MACHINE

REVISE_ASM_THREADS=$((2*$REVISE_TOTAL_GPUS))
echo REVISE_ASM_THREADS=$REVISE_ASM_THREADS

# Input dataset name
if [ -z "$REVISE_DATASET" ]
then
    read -p "Dataset name [$REVISE_DATASET_DEFAULT]: " -i 1 REVISE_DATASET
    REVISE_DATASET=${REVISE_DATASET:-$REVISE_DATASET_DEFAULT}
fi
echo REVISE_DATASET=$REVISE_DATASET

# Download and preprocess source dataset, which results in a ReVisE dataset
revise_prepare_dataset.sh $REVISE_DATASET

# Read dataset description
source $REVISE_DATASET/description.sh

# Prepare custom ReVisE Web server problem list files for testing the specified dataset
revise_webdata.js install test_datasets.json
revise_install_dataset.sh $REVISE_DATASET

# Prepare custom ReVisE Web server configuration files for testing
revise_hw_config.js install test_hw.json
revise_hw_config.js set nodes 1
revise_hw_config.js set assemble_threads_per_node $REVISE_ASM_THREADS
revise_hw_config.js set measure_time true



# Measure performance
echo Starting performance measurement for dataset ${REVISE_DATASET}

# Create directory for performance log files.
LOGDIR=log/$REVISE_MACHINE/$REVISE_DATASET
mkdir -p $LOGDIR

REVISE_GPUS=1
TOTAL_MEASUREMENT_COUNT=$(( REVISE_TOTAL_GPUS * 3 ))
CURRENT_MEASUREMENT_NUMBER=1
while (( REVISE_GPUS <= REVISE_TOTAL_GPUS ))
do
    for REVISE_WORKERS_PER_GPU in 1 2 4; do

        REVISE_WORKERS=$(( REVISE_WORKERS_PER_GPU * REVISE_GPUS ))

        # Specify hardware configuration to use by ReVisE
        revise_hw_config.js set gpus_per_node $REVISE_GPUS
        revise_hw_config.js set render_threads_per_node $REVISE_WORKERS

        # Prepare Web browser, if necessary
        if [ -z "$REVISE_WEBCLIENT_INITIALIZED" ]
        then
            echo "Prepare Web browser for measuring ReVisE performance.
Please follow instructions below."
            read -p "Press enter to continue"
            echo "-------- 8< --------"
            echo "Web server will be started now. When it starts for the first time,
necessary dependencies are downloaded and installed,
and two ReVisE modules that interoperate with node.js are built.
This process takes a few seconds. When the server is ready, it displays the message

Starting web server.

As soon as you see this message, do the following.
- Open Web browser (we use Chrome and Firefox, other browser should work too).
- type localhost:3000 in the address bar and press Enter.
- Maximize browser window.
- Select problem -> choose dataset name (${REVISE_DATASET}).
- Select field -> ${REVISE_DATASET_FIELD_NAME}.
- Settings -> Reset data; wait a few seconds.
- Settings -> set fovY to ${REVISE_DATASET_CAMERA_FOVY}; Fields are shown as relative values; Time interval, ms = 1; Ok.
- Select Visualization mode -> MIP.
- Move to the left the slider below the visualization mode selector.
- Once you have done the above steps, go to the terminal and press Ctrl+C (ONE TIME!).
  Note: Do not close the browser until the performance measurement finishes!
You may want to copy the above instructions to a text file
or take a screenshot, because the page will scroll soon.
"
            read -p "Press enter now to start the Web server"
            revise_web.sh
            echo "-------- 8< --------"
            read -p "Now go the Web browser and refresh the window; then return to the terminal and press Enter."
            echo "ReVisE Web server will now start again. After that, do the following.
- Go to the Web browser.
- Refresh browser window.
- Colormap settings -> Ok.
- Go to terminal and press Ctrl+C (ONE TIME!) to stop the Web server.
You may want to copy the above instructions to a text file
or take a screenshot, because the page will scroll soon.
"
            read -p "Press enter now to start the Web server"
            revise_web.sh
            echo "-------- 8< --------"
            echo "Web browser configuration is complete."
            read -p "Now go the Web browser and refresh the window; then return to the terminal and press Enter."
            REVISE_WEBCLIENT_INITIALIZED=true
        fi

        # Remove timestamp log if it exists
        rm -f $REVISE_ROOT_DIR/src/webserver/VsRendererDrawTimestamps.log

        # Do performance measurement for specific configuration
        echo "-------- 8< --------"
        echo "Performance measurement for REVISE_GPUS=$REVISE_GPUS, REVISE_WORKERS=$REVISE_WORKERS"
        echo "Web server will start now. Then please follow instructions below.
- Go to the Web browser
- Refresh browser window
- Wait till \"Frame level: $REVISE_DATASET_MAX_LEVEL\" is reported in the lower left corner of the window.
  First time it might take about $REVISE_DATASET_WARMINGUP_TIME s, in the minimal HW configuration.
- Press button Y|_X 10 times, each time waiting till \"Frame level: $REVISE_DATASET_MAX_LEVEL\" is reported.
- Go to terminal and press Ctrl+C (ONE TIME!) to stop the Web server.
You may want to copy the above instructions to a text file
or take a screenshot, because the page will scroll soon."
        read -p "Press enter now to start the Web server"
        revise_web.sh
        echo "-------- 8< --------"
        # Move the timestamp log file to the log directory:
        mv $REVISE_ROOT_DIR/src/webserver/VsRendererDrawTimestamps.log $LOGDIR/VsRendererDrawTimestamps-$REVISE_GPUS-$REVISE_WORKERS-$REVISE_ASM_THREADS.log
        echo "Measurement $CURRENT_MEASUREMENT_NUMBER of $TOTAL_MEASUREMENT_COUNT is complete."
        read -p "Now go the Web browser and refresh the window; then return to the terminal and press Enter."

        # Increase measurement number
        CURRENT_MEASUREMENT_NUMBER=$(( CURRENT_MEASUREMENT_NUMBER + 1 ))
    done

    # Increase number of GPUs (x2 or set to the number of total available GPUs)
    if (( REVISE_GPUS < REVISE_TOTAL_GPUS ))
    then
        REVISE_GPUS=$((REVISE_GPUS*2))
        if (( REVISE_GPUS > REVISE_TOTAL_GPUS ))
        then
            REVISE_GPUS=$REVISE_TOTAL_GPUS
        fi
    else
        REVISE_GPUS=$((REVISE_GPUS*2))
    fi
done

echo "-------- 8< --------"
echo "Performance measurement is complete, processing results."

# Process measurement results
revise_process_render_perf_logs.sh $LOGDIR

# Print results
cat $LOGDIR/total_avg.tsv
