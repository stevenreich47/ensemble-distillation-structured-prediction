#! /usr/bin/env bash

# Define the following environment variables:
#
# JOBS_DIR - directory with trained models
# OUTPUT_DIR - where to save teacher confidences
# DATA_DIR - path to preprocessed wmt16 data

set -e
set -u

if [ $# -lt 2 ]; then
   echo "Usage: ${0} JOB_NAME SEED [FLAGS]"
   exit
fi

JOB_NAME=${1}
SEED=${2}
shift
shift


if [ ! -d "${DATA_DIR}" ]; then
    echo "${DATA_DIR} does not exist"
    exit
fi

mkdir -p ${JOB_DIR}
JOB_SCRIPT=${JOB_DIR}/job.sh

TRAIN="fairseq-train"

# Write training script
cat >${JOB_SCRIPT} <<EOL
#$ -cwd
#$ -V
#$ -w e
#$ -N ${JOB_NAME}
#$ -m bea
#$ -j y
#$ -o ${JOB_DIR}/out
#$ -e ${JOB_DIR}/err

# Stop on error
set -e
set -u
set -f

module load cuda10.1/toolkit
module load cuda10.1/blas
module load cudnn/7.6.3_cuda10.1
module load nccl/2.4.7_cuda10.1
export MKL_SERVICE_FORCE_INTEL=1

fairseq-train \
    ${DATA_DIR} \
    --source-lang de --target-lang en \
    --arch transformer_wmt_en_de --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.0007 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
    --weight-decay 0.0 \
    --dropout 0.1 \
    --criterion cross_entropy \
    --no-progress-bar \
    --save-dir ${JOB_DIR} \
    --seed ${SEED} \
    --max-tokens 4096 \
    --fp16 \
    --keep-last-epochs 10 \
    --update-freq ${UPDATE_FREQ} \
    --max-update ${MAX_UPDATE} $@

EOL

chmod a+x ${JOB_SCRIPT}
${JOB_SCRIPT}

# eof
