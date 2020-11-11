#! /usr/bin/env bash

set -e
set -u

# Define the following environment variables:
#
# JOBS_DIR - directory with trained models 
# DATA_DIR - path to preprocessed wmt16 data

if [ $# -lt 2 ]; then
    echo "Usage: ${0} <SPLIT> <JOBS>"
    ls ${JOBS_DIR}
    exit
fi

SPLIT=$1
shift

CHECKPOINTS=`python join_ensemble_path.py ${JOBS_DIR} $@`

echo "Checkpoints: ${CHECKPOINTS}"

VALIDATE=`realpath ../../validate.py`

python ${VALIDATE} ${DATA_DIR} \
       --measure-calibration \
       --source-lang de \
       --target-lang en \
       --task translation \
       --criterion cross_entropy \
       --valid-subset ${DATA_DIR}/${SPLIT} \
       --path ${CHECKPOINTS} \
       --max-tokens 4096 \
       --num-workers 10

# eof
