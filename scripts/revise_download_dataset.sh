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

# Download dataset if necessary
if [ ! -f ${REVISE_DATASET}/data_downloaded ]
then
    echo "Downloading dataset ${REVISE_DATASET}"

    # Download and extract the dataset
    DATASET_FILENAME=${REVISE_DATASET}_source.tar.gz
    DOWNLOAD_LINK=https://ftp.mpksoft.ru/revise_datasets/$REVISE_DATASET/$DATASET_FILENAME
    if [ -f "$DATASET_FILENAME" ]
    then
        # Continue interrupted download
        curl $DOWNLOAD_LINK -C $(stat --format=%s prepare_cascade "$DATASET_FILENAME") --output $DATASET_FILENAME
    else
        # Download
        curl $DOWNLOAD_LINK --output $DATASET_FILENAME
    fi
    echo >${REVISE_DATASET}/data_downloaded
fi

# Extract dataset if necessary
if [ ! -f ${REVISE_DATASET}_extracted ]
then
    tar -zxf $DATASET_FILENAME
    echo >${REVISE_DATASET}_extracted
    # rm $DATASET_FILENAME
fi
