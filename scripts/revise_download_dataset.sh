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

DATASET_FILENAME="${REVISE_DATASET}_source.tar.gz"
DATASET_RELPATH="$REVISE_DATASET/$DATASET_FILENAME"

# Download dataset if necessary
if [ ! -f "${REVISE_DATASET}/data_downloaded" ]
then
    echo "Downloading dataset ${REVISE_DATASET}"

    # Download and extract the dataset
    DOWNLOAD_LINK="https://ftp.mpksoft.ru/revise_datasets/$DATASET_RELPATH"
    if [ -f "$DATASET_RELPATH" ]
    then
        # Continue interrupted download
        curl "$DOWNLOAD_LINK" -C $(stat --format=%s "$DATASET_RELPATH") --output "$DATASET_RELPATH"
    else
        # Download
        curl "$DOWNLOAD_LINK" --output "$DATASET_RELPATH"
    fi
    echo >"${REVISE_DATASET}/data_downloaded"
fi

# Extract dataset if necessary
if [ ! -f "${REVISE_DATASET}/data_extracted" ]
then
    echo "Extracting dataset ${REVISE_DATASET}"
    tar -zxf "$DATASET_RELPATH" -C "$REVISE_DATASET"
    echo >"${REVISE_DATASET}/data_extracted"
    # rm $DATASET_RELPATH
fi
