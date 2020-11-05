#! /usr/bin/env bash

set -e
set -u

# Define the following environment variables:
#
# JOBS_DIR - directory with trained models
# OUTPUT_DIR - where to save teacher confidences
# DATA_DIR - path to preprocessed wmt16 data

# Where we save the confidences
mkdir -p ${OUTPUT_DIR}

if [ $# -lt 2 ]; then
    echo "Usage: ${0} <TOPK> <CKPT> <PATHS>"
    ls ${JOBS_DIR}
    exit
fi

echo "Additional args: $@"

TOP_K=${1}
CKPT=${2}
shift
shift

CHECKPOINTS=`python join_ensemble_path.py --checkpoint ${CKPT} ${JOBS_DIR} $@`
echo "Checkpoints: ${CHECKPOINTS}"

VALIDATE=`realpath ../../validate.py`
OUTPUT_FILE="$@_${TOP_K}_${CKPT}.h5"
OUTPUT_FILE=${OUTPUT_DIR}/"${OUTPUT_FILE// /_}"

echo "Output file: ${OUTPUT_FILE}"
echo "Top K: ${TOP_K}"

python ${VALIDATE} ${DATA_DIR} \
       --task translation \
       --criterion cross_entropy \
       --valid-subset ${DATA_DIR}/train \
       --full-dist-path ${OUTPUT_FILE} \
       --path ${CHECKPOINTS} \
       --max-tokens 2048 \
       --print-full-dist \
       --dist-top-k ${TOP_K} \
       --fp16 \
       --num-workers 10

# eof
