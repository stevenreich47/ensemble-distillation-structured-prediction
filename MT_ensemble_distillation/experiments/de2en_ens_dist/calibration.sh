#! /usr/bin/env bash

set -e
set -u

# Define the following environment variables:
#
# DATA_DIR - path to preprocessed wmt16 data

if [ $# -lt 3 ]; then
    echo "Usage: ${0} <SPLIT> <JOBS_DIR> <JOBS>"
    ls ${JOBS_DIR}
    exit
fi

SPLIT=$1
JOBS_DIR=$2
shift
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
