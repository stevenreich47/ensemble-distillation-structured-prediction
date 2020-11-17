set -u

#$ -q gpu.q
#$ -cwd
#$ -V
#$ -l h_rt=3:00:00,num_proc=10,mem_free=10G,gpu=1

module load cuda10.0/toolkit
module load cudnn/7.5.0_cuda10.0
module load nccl/2.4.2_cuda10.0

OUTPUT_DIR=${GRIFFIN_EXP_DIR}/scale/bert/bert_l2

if [ ! -d ${OUTPUT_DIR} ]; then
  mkdir ${OUTPUT_DIR}
fi

echo "training model..."
python ${GRIFFIN_REPO_DIR}/scripts/train_from_features.py fit \
  --train-tfrecord-path ${GRIFFIN_EXP_DIR}/scale/bert_embed/ru_with_alignment/train.tf \
  --label-map-path ${GRIFFIN_EXP_DIR}/scale/bert_embed/ru_with_alignment/label_map.pickle \
  --hold-out-fraction 0.1 \
  --early-stop-patience 3 \
  --data-format bert_tokens_with_words_cased \
  --model bert_lstm_crf \
  --hparams bert_rubert_bi_crf_adafactor_clipped_aligned_l2 \
  --model-path ${OUTPUT_DIR}/checkpoints \
  --train-batch-size 16 \
  --warm-start-from ${SCALE_DIR}/models/bert/rubert_cased_L-12_H-768_A-12_v1/bert_model.ckpt \
  --warm-start-vars bert* \
  --global-max-len 512 \
  --output-file ${OUTPUT_DIR}/predictions.txt \
  --output-confidences ${OUTPUT_DIR}/confidences.txt

echo "making predictions on test data..."
python ${GRIFFIN_REPO_DIR}/scripts/train_from_features.py predict \
  --test-tfrecord-path ${GRIFFIN_EXP_DIR}/scale/bert_embed/ru_with_alignment/dev.tf \
  --label-map-path ${GRIFFIN_EXP_DIR}/scale/bert_embed/ru_with_alignment/label_map.pickle \
  --data-format bert_tokens_with_words_cased \
  --hparams bert_rubert_bi_crf_adafactor_clipped_aligned_l2 \
  --model-path ${OUTPUT_DIR}/checkpoints \
  --output-file ${OUTPUT_DIR}/predictions.txt \
  --output-confidences ${OUTPUT_DIR}/confidences.txt \
  --model bert_lstm_crf

echo "Evaluating predictions on test data..."
${GRIFFIN_REPO_DIR}/scripts/conlleval <${OUTPUT_DIR}/predictions.txt >${OUTPUT_DIR}/eval_results.txt

echo "All done! Results are in: ${OUTPUT_DIR}/eval_results.txt"
