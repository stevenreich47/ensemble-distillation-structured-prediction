#! /usr/bin/env bash

set -e
set -u

# NOTE: Default learning rate: --lr 0.0005
#
# NOTE: Dropout is disabled
#
# NOTE: We initialize from an average checkpoint rather than the last.
#
# NOTE: float16 is disabled

# NOTE: LS = 0 -> no label smoothing

# --- SYSTEM ---
N_GPU=4  # args must be adjusted below if this is changed
NUM_PROC=40
GPU_TYPE=2080
MEM=12G
HOURS=48

# --- BATCHING ---
UPDATE_FREQ=2
MAX_TOKENS=4096
WARMUP_UPDATE=500
SAVE_INTERVAL_UPDATES=2000

TEACHER_DIR=/expscratch/nandrews/nmt/fairseq/jobs/de2en/teachers

if [ $# -lt 9 ]; then
   echo "Usage: ${0} JOB_NAME TOPK TEMP WEIGHT INIT MAX_UPDATE DIVERGENCE LR LS TEACHERS"
   exit
fi

JOB_NAME=${1}
TOPK=${2}
T=${3}
WEIGHT=${4}  # teacher weight
INIT_JOB=${5}
MAX_UPDATE=${6}
DIVERGENCE=${7}
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

DATA_DIR=/expscratch/nandrews/nmt/fairseq/data/wmt16_de_en_bpe32k
if [ ! -d "${DATA_DIR}" ]; then
    echo "${DATA_DIR} does not exist"
    exit
fi

TEACHER_FILE=`python join_ensemble_path.py ${TEACHER_DIR} ${TEACHERS}`
echo "${TEACHER_FILE}"

INIT_DIR=/expscratch/nandrews/nmt/fairseq/jobs/de2en
INIT_FILE="${INIT_DIR}/${INIT_JOB}/checkpoint_avg.pt"
if [ ! -f "${INIT_FILE}" ]; then
    echo "${INIT_FILE} not found"
    ls -l ${INIT_DIR}
    exit
fi
echo "Init file: ${INIT_FILE}"

JOB_DIR=/expscratch/${USER}/nmt/fairseq/jobs/de2en_ens_distill/${JOB_NAME}_${INIT_JOB}_${DIVERGENCE}_${T}_${TOPK}_${WEIGHT}_${MAX_UPDATE}_${LR}_${LS}_${TEACHERS}
JOB_DIR="${JOB_DIR// /_}"
echo "Job dir: ${JOB_DIR}"
mkdir -p ${JOB_DIR}
JOB_SCRIPT=${JOB_DIR}/job.sh

echo "${JOB_DIR}"
TRAIN="fairseq-train"

#    --ddp-backend=no_c10d

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
    --source-lang de \
    --target-lang en \
    --task translation_with_teacher \
    --reset-optimizer \
    --reset-lr-scheduler \
    --reset-dataloader \
    --restore-file ${INIT_FILE} \
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
    --dropout 0.0 \
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
QSUB_CMD="qsub -q gpu.q@@${GPU_TYPE} -l gpu=${N_GPU},mem_free=${MEM},h_rt=${HOURS}:00:00,num_proc=${NUM_PROC} ${JOB_SCRIPT}"
echo ${QSUB_CMD}
${QSUB_CMD}
#bash ${JOB_SCRIPT}

# eof
