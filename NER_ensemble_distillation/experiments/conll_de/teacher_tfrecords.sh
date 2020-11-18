#! /usr/bin/env bash
set -euo pipefail

# This assumes you have the following environment variables defined:
#
#   MBERT_DIR: directory containing vocab and ckpt for m-BERT
#   NER_REPO_DIR: directory with the project repository
#   NER_EXP_DIR: directory for storing experiment artifacts
#   NER_DATA_DIR: directory containing iob2 splits for CoNLL 2003

if [ $# -lt 1 ]; then
    echo "Usage: ${0} MODEL_TYPE"
    echo "MODEL_TYPE may be 'iid' or 'crf'"
    exit
fi

source "${NER_REPO_DIR}"/experiments/grid-env.sh

MODEL_TYPE=${1}

OUTPUT_DIR=${NER_EXP_DIR}/conll_de
TRAIN=${NER_DATA_DIR}/conll.de.train.iob2
TEACHER=${OUTPUT_DIR}/${MODEL_TYPE}_ensemble/teachers.txt
DEV=${NER_DATA_DIR}/conll.de.dev.iob2
TEST=${NER_DATA_DIR}/conll.de.test.iob2
LABEL_MAP=${OUTPUT_DIR}/label_map.pickle
DATA_FORMAT=bert_tokens_cased_with_teacher_dists
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

echo "Converting ${TEACHER} to TFRecords: ${OUTPUT_DIR}/${MODEL_TYPE}_teacher.tf"
${SCRIPT} use_teacher_dists=True conll=${TEACHER} --output ${OUTPUT_DIR}/${MODEL_TYPE}_teacher.tf

echo "Converting ${DEV} to TFRecords: ${OUTPUT_DIR}/t_dev.tf"
${SCRIPT} conll=${DEV} --output ${OUTPUT_DIR}/t_dev.tf

echo "Converting ${TEST} to TFRecords: ${OUTPUT_DIR}/t_test.tf"
${SCRIPT} conll=${TEST} --output ${OUTPUT_DIR}/t_test.tf
