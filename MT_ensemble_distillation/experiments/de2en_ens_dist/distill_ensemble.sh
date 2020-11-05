#! /usr/bin/env bash

# Define the following environment variables:
#
# JOBS_DIR - directory with trained models
# DATA_DIR - path to preprocessed wmt16 data
# TEACHER_DIR - dir with memoized teacher confidences

set -e
set -u

# --- BATCHING ---
UPDATE_FREQ=2
MAX_TOKENS=4096
WARMUP_UPDATE=5000
SAVE_INTERVAL_UPDATES=2000

TEACHER_DIR=/expscratch/nandrews/nmt/fairseq/jobs/de2en/teachers

if [ $# -lt 9 ]; then
   echo "Usage: ${0} JOB_NAME TOPK TEMP WEIGHT MAX_UPDATE DIVERGENCE DROPOUT LR LS TEACHERS"
   exit
fi

JOB_NAME=${1}
TOPK=${2}
T=${3}
WEIGHT=${4}  # teacher weight
MAX_UPDATE=${5}
DIVERGENCE=${6}
DROPOUT=${7}
LR=${8}
LS=${9}
shift
shift
shift
shift
shift
shift
shift
shift
shift
TEACHERS="$@"

echo "Temperature: ${T}"
echo "Distillation loss weight: ${WEIGHT}"
echo "Divergence: ${DIVERGENCE}"
echo "Teachers: ${TEACHERS}"
echo "Max update: ${MAX_UPDATE}"
echo "Learning rate: ${LR}"
echo "Label smoothing: ${LS}"
echo "Dropout: ${DROPOUT}"

if [ ! -d "${DATA_DIR}" ]; then
    echo "${DATA_DIR} does not exist"
    exit
fi

TEACHER_FILE=`python join_ensemble_path.py ${TEACHER_DIR} ${TEACHERS}`
echo "Teacher: ${TEACHER_FILE}"

JOB_DIR=${JOBS_DIR}/${JOB_NAME}_FROM_SCRATCH_${DIVERGENCE}_${T}_${TOPK}_${WEIGHT}_${MAX_UPDATE}_${LR}_${LS}_${DROPOUT}_${TEACHERS}
JOB_DIR="${JOB_DIR// /_}"
echo "Job dir: ${JOB_DIR}"
mkdir -p ${JOB_DIR}
JOB_SCRIPT=${JOB_DIR}/job.sh

echo "${JOB_DIR}"
TRAIN="fairseq-train"

# Write training script
cat >${JOB_SCRIPT} <<EOL
fairseq-train \
    ${DATA_DIR} \
    --source-lang de \
    --target-lang en \
    --task translation_with_teacher \
    --reset-optimizer \
    --reset-lr-scheduler \
    --reset-dataloader \
    --teacher-pred ${TEACHER_FILE} \
    --teacher-top-k ${TOPK} \
    --distill-loss-type combined \
    --distill-divergence ${DIVERGENCE} \
    --distill-temperature ${T} \
    --label-smoothing ${LS} \
    --teacher-weight ${WEIGHT} \
    --arch transformer_wmt_en_de --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr ${LR} --lr-scheduler inverse_sqrt \
    --warmup-updates ${WARMUP_UPDATE} \
    --warmup-init-lr 1e-07 \
    --weight-decay 0.0 \
    --dropout ${DROPOUT} \
    --criterion distillation_cross_entropy \
    --no-progress-bar \
    --save-dir ${JOB_DIR} \
    --tensorboard-logdir ${JOB_DIR}\tensorboard \
    --max-tokens ${MAX_TOKENS} \
    --keep-last-epochs 10 \
    --update-freq ${UPDATE_FREQ} \
    --save-interval-updates ${SAVE_INTERVAL_UPDATES} \
    --keep-interval-updates 5 \
    --max-update ${MAX_UPDATE} \
    --fp16

EOL

chmod a+x ${JOB_SCRIPT}
bash ${JOB_SCRIPT}

# eof
