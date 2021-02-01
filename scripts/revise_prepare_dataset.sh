#!/bin/bash

set -e

# Check arguments
REVISE_DATASET=$1
if [ -z "${REVISE_DATASET}" ]
then
    echo "Please specify the revise dataset name in first argument."
    echo "Available datasets names are: cascade, hump, mwave, sphere_1M."
    exit 1
fi

# Set up environment, if necessary
[ -z "$(which s3dmm_prep)" ] && source $( dirname "${BASH_SOURCE[0]}" )/env.sh

# Create dataset direcory, if necessary
mkdir -p ${REVISE_DATASET}

# Download dataset description if necessary
if [ ! -f ${REVISE_DATASET}/description_ready ]
then
    echo "Downloading dataset ${REVISE_DATASET} description"

    # Download and extract dataset description
    curl https://ftp.mpksoft.ru/revise_datasets/${REVISE_DATASET}/${REVISE_DATASET}_description.tar.gz | tar -zx
    echo >${REVISE_DATASET}/description_ready
fi

# Prepare dataset, if necessary
if [ ! -f ${REVISE_DATASET}/revise_ready ]
then
    if [ -f ${REVISE_DATASET}/generate.sh ]
    then
        ${REVISE_DATASET}/generate.sh
    else
        revise_download_dataset.sh ${REVISE_DATASET}
        ${REVISE_DATASET}/prepare.sh
    fi
    echo >${REVISE_DATASET}/revise_ready
fi

echo "Dataset ${REVISE_DATASET} is ready for visualization"
