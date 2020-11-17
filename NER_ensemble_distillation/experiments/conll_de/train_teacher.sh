#! /usr/bin/env bash

set -e
set -f
set -u

module load cuda10.1/toolkit
module load cudnn/7.6.3_cuda10.1
module load nccl/2.4.7_cuda10.1

if [ $# -lt 2 ]; then
    echo "Usage: ${0} MODEL_TYPE MODEL_NUM"
    echo "MODEL_TYPE can be 'crf' or 'iid'"
    echo "0 <= MODEL_NUM < 9"
    exit
fi

GPU_DETECT=${GRIFFIN_REPO_DIR}/scripts/gpus_available.py
GPU_DETECT_CMD="python ${GPU_DETECT}"

MODEL_TYPE=${1}
MODEL_NUM=${2}

TRAINER=${GRIFFIN_REPO_DIR}/scripts/train_from_features.py
CMD="python ${TRAINER}"
DATA_DIR=${GRIFFIN_EXP_DIR}/conll_de
TRAIN=${DATA_DIR}/train.tf
DEV=${DATA_DIR}/dev.tf
TEST=${DATA_DIR}/test.tf
LABEL_MAP=${DATA_DIR}/label_map.pickle
OUTPUT_DIR=${DATA_DIR}/${MODEL_TYPE}_${MODEL_NUM}
MODEL_DIR=${OUTPUT_DIR}/checkpoints
EVAL=${GRIFFIN_REPO_DIR}/scripts/conlleval
DATA_FORMAT=bert_tokens_with_words_cased
if [ ${MODEL_TYPE} == 'crf' ]; then
    MODEL=bert_lstm_crf
    HPARAMS=bert_multi_bi_crf_adafactor_clipped_aligned_nodecay
fi

if [ ${MODEL_TYPE} == 'iid' ]; then
    MODEL=bert_iid
    HPARAMS=bert_unfrozen_iid_multi
fi
BERT_CHECKPOINT=${MBERT_DIR}/bert_model.ckpt
BERT_VARS="bert*"
TRAIN_BATCH_SIZE=8
EVAL_BATCH_SIZE=8

DIM=`python ${GRIFFIN_REPO_DIR}/scripts/output_dimension.py --label-map=${LABEL_MAP}`

echo "Checkpoint directory: ${MODEL_DIR}"

if [ ! -f ${LABEL_MAP} ]; then
    echo "Missing label map"
    exit
fi

if [ ! -f ${TRAIN} ]; then
    echo "Missing training records:"
    ls ${DATA_DIR}
    exit
fi

if [ -d ${MODEL_DIR} ]; then
    echo "Removing old checkpoints..."
    rm -rf ${MODEL_DIR}
fi

echo "Checking for GPUs"
${GPU_DETECT_CMD}


echo "Training the model."
${CMD} fit \
       --train-tfrecord-path ${TRAIN} \
       --dev-tfrecord-path ${DEV} \
       --label-map-path ${LABEL_MAP} \
       --hold-out-fraction 0.1 \
       --early-stop-patience 3 \
       --data-format ${DATA_FORMAT} \
       --model ${MODEL} \
       --hparams ${HPARAMS} \
       --model-path ${MODEL_DIR} \
       --train-batch-size ${TRAIN_BATCH_SIZE} \
       --warm-start-from ${BERT_CHECKPOINT} \
       --warm-start-vars ${BERT_VARS} \
       --custom-split ${MODEL_NUM} \
       --hparams-str "output_dim=${DIM}"

echo "Get predictions on train data."
${CMD} predict \
       --test-tfrecord-path ${TRAIN} \
       --label-map-path ${LABEL_MAP} \
       --data-format ${DATA_FORMAT} \
       --output-file ${OUTPUT_DIR}/train_predictions.txt \
       --model ${MODEL} \
       --hparams ${HPARAMS} \
       --model-path ${MODEL_DIR} \
       --eval-batch-size ${EVAL_BATCH_SIZE} \
       --hparams-str "output_dim=${DIM}" \
       --report-confidences True \
       --full-dist True \
       --bio-dist True \
       --output-confidences ${OUTPUT_DIR}/bio_train_confidences.txt

echo "Get predictions on dev data."
${CMD} predict \
       --test-tfrecord-path ${DEV} \
       --label-map-path ${LABEL_MAP} \
       --data-format ${DATA_FORMAT} \
       --output-file ${OUTPUT_DIR}/dev_predictions.txt \
       --model ${MODEL} \
       --hparams ${HPARAMS} \
       --model-path ${MODEL_DIR} \
       --eval-batch-size ${EVAL_BATCH_SIZE} \
       --hparams-str "output_dim=${DIM}"

echo "Get predictions on test data."
${CMD} predict \
       --test-tfrecord-path ${TEST} \
       --label-map-path ${LABEL_MAP} \
       --data-format ${DATA_FORMAT} \
       --output-file ${OUTPUT_DIR}/test_predictions.txt \
       --model ${MODEL} \
       --hparams ${HPARAMS} \
       --model-path ${MODEL_DIR} \
       --eval-batch-size ${EVAL_BATCH_SIZE} \
       --hparams-str "output_dim=${DIM}" \
       --report-confidences True \
       --full-dist True \
       --output-confidences ${OUTPUT_DIR}/test_confidences.txt

# eof
