#!/bin/bash

set -e

# Check arguments
REVISE_DATASET="$1"
if [ -z "$REVISE_DATASET" ]
then
    echo "Please specify the revise dataset name in first argument."
    echo "Available datasets names are: cascade, hump, mwave, sphere_1M."
    exit 1
fi

shift
# Note: Remaining arguments are supplied to description/prepare/generate scripts.


# Set up environment, if necessary
source $REVISE_ROOT_DIR/scripts/env.sh

# Read dataset description
if [ -f $REVISE_DATASET/description.sh ]
then
    source $REVISE_DATASET/description.sh "$@"
else
    echo "Cannot find file $REVISE_DATASET/description.sh"
    exit 1
fi

# Check that REVISE_DATASET_PATH is set
if [ -z "$REVISE_DATASET_PATH" ]
then
    echo "REVISE_DATASET_PATH environment variablle is not set by $REVISE_DATASET/description.sh"
    exit 1
fi

# Check that REVISE_DATASET_NAME is set
if [ -z "$REVISE_DATASET_NAME" ]
then
    echo "REVISE_DATASET_NAME environment variablle is not set by $REVISE_DATASET/description.sh"
    exit 1
fi

if [ ! -f "${REVISE_DATASET}/${REVISE_DATASET_NAME}#revise_ready" ]
then
    echo "${REVISE_DATASET}/${REVISE_DATASET_NAME} is not ready, run revise_prepare_dataset.sh"
    exit 1
fi

# Add dataset to the list available to the Web server
[ ! -z $(revise_webdata.js list |grep "^$REVISE_DATASET_NAME$") ] && revise_webdata.js remove_by_title $REVISE_DATASET_NAME
revise_webdata.js add "$REVISE_DATASET_PATH" "$REVISE_DATASET_NAME"
