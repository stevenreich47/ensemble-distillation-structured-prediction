#! /usr/bin/env bash
set -euo pipefail

# This assumes you have the following environment variables defined:
#
#   MBERT_DIR: directory containing vocab and ckpt for m-BERT
#   NER_REPO_DIR: directory containing this repository
#   NER_EXP_DIR: directory for storing experiment artifacts
#   NER_DATA_DIR: directory containing processed conll data

source "${NER_REPO_DIR}"/experiments/grid-env.sh

OUTPUT_DIR=${NER_EXP_DIR}/conll_de
TRAIN=${NER_DATA_DIR}/conll.de.train.iob2
DEV=${NER_DATA_DIR}/conll.de.dev.iob2
TEST=${NER_DATA_DIR}/conll.de.test.iob2
LABEL_MAP=${OUTPUT_DIR}/label_map.pickle
DATA_FORMAT=bert_tokens_with_words_cased
VOCAB=${MBERT_DIR}/vocab.txt

MAX_SENTENCE_LEN=510

SCRIPT="${NER_REPO_DIR}/scripts/write_tfrecords.py
    --data-format ${DATA_FORMAT}
    --max-sentence-len ${MAX_SENTENCE_LEN}
    --parser conll_subwords_with_alignment
    --parser-args label_map=${LABEL_MAP}
                  vocab=${VOCAB}
                  max_sentence_len=${MAX_SENTENCE_LEN}"


if [ ! -d ${OUTPUT_DIR} ]; then
    mkdir -p ${OUTPUT_DIR}
fi

echo "Writing label map to: ${LABEL_MAP}"
${NER_REPO_DIR}/scripts/build_label_map.py --input-paths "${TRAIN},${DEV},${TEST}" --output-path ${LABEL_MAP}

echo "Converting ${TRAIN} to TFRecords: ${OUTPUT_DIR}/train.tf"
${SCRIPT} conll=${TRAIN} --output ${OUTPUT_DIR}/train.tf

echo "Converting ${DEV} to TFRecords: ${OUTPUT_DIR}/dev.tf"
${SCRIPT} conll=${DEV} --output ${OUTPUT_DIR}/dev.tf

echo "Converting ${TEST} to TFRecords: ${OUTPUT_DIR}/test.tf"
${SCRIPT} conll=${TEST} --output ${OUTPUT_DIR}/test.tf
