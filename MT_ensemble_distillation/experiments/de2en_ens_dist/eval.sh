#! /usr/bin/env bash

# Define the following environment variables:
#
# JOBS_DIR - directory with trained models
# OUTPUT_DIR - where to save teacher confidences
# DATA_DIR - path to preprocessed wmt16 data

set -e
set -u

if [ $# -lt 1 ]; then
    echo "Usage: ${0} JOB_NAME"
    exit
fi

JOB_NAME=${1}
JOB_DIR=${JOBS_DIR}/${JOB_NAME}

AVG=`realpath ../../scripts/average_checkpoints.py`

python ${AVG} \
    --inputs ${JOB_DIR} \
    --num-epoch-checkpoints 10 \
    --output /tmp/avg_checkpoint.pt

# Note: without `--quiet`, this will print the translations and corresponding
# sequence- and token-level scores.
GEN=/tmp/gen.out
fairseq-generate \
    ${DATA_DIR} \
    --source-lang de \
    --target-lang en \
    --path /tmp/avg_checkpoint.pt \
    --beam 4 --lenpen 0.6 --remove-bpe > ${GEN}

tail -n 3 ${GEN}

# See: https://github.com/pytorch/fairseq/issues/346

SYS="${GEN}.sys"
REF="${GEN}.ref"

grep ^H $GEN | cut -f3- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > $SYS
grep ^T $GEN | cut -f2- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > $REF

fairseq-score --sys ${SYS} --ref ${REF}

# eof
