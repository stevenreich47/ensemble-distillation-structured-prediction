# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
# Copyright 2019 Johns Hopkins University
# Copyright 2020 Johns Hopkins University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An object wrapper for `train_from_features`

This allows for more modular operations to be
performed with training, such as iteratively training
"""

# pylint: disable=import-outside-toplevel
import os
import pickle
import time
from os import rename
from pathlib import Path
from shutil import rmtree
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow as tf2
from absl import logging
from typing_extensions import Literal

from griffin.confidences import est_span_conf
from griffin.data_format import DataFormat
from griffin.crf import span_confidence
from griffin.crf import token_confidence
from griffin.features import Features
from griffin.features import Predictions
from griffin.gpu import get_run_config
from griffin.hparam import HParams
from griffin.hparams import hparams_maybe_override
from griffin.hparams import pretty_print_hparams
from griffin.parser_utils import agg_first
from griffin.parser_utils import agg_mode_type_first_prefix
from griffin.parser_utils import collapse_subword_values
from griffin.registry import Registries
from griffin.conlleval import report
from griffin.conlleval import evaluate
from griffin.conlleval import full_metrics
from griffin.tfrecord import check_tfrecord_input
from griffin.tfrecord import number_of_records
from griffin.tfrecord import sanity_check_tfrecord

#logging.set_verbosity(logging.INFO)
#absl.logging.set_verbosity(absl.logging.INFO)

def _tfrecords_from_path(data_type: str, path: str) -> tf.data.TFRecordDataset:
  if not path:
    raise ValueError(f'{data_type} tfrecord path must be provided')
  return tf.data.TFRecordDataset(tf.data.Dataset.list_files(path))


class Trainer:
  """Arguments to train_from_features ported here as member variables"""

  def __init__(
      self,
      sanity_checks=True,
      eval_throttle_secs=60,
      train_tfrecord_path=None,
      hold_out_fraction=None,
      test_tfrecord_path=None,
      data_format='fasttext',
      model='feature_bi_crf_model',
      hparams='bi_crf_default_9',
      hparams_str=None,
      hparams_json=None,
      label_map_path='label_map.pickle',
      model_path='checkpoints',
      warm_start_from=None,
      warm_start_vars='*',
      output_file='predictions.txt',
      output_confidences='confidences.txt',
      conf_method='average',
      report_confidences=False,
      full_dist=False,
      bio_dist=False,
      logit_dist=False,
      temperature=None,
      save_checkpoints_steps=10000,
      keep_checkpoint_hours=10000,
      train_epochs=100,
      max_train_steps=10000,
      repeat=1,
      min_epochs=2,
      early_stop_patience=3,
      train_batch_size=64,
      eval_batch_size=64,
      shuffle_buffer_size=1000,
      random_seed=None,
      n_gpu=0,
      n_cpu=6,
      token_calibrate=False,
      logging_names="learning_rate",
      dev_tfrecord_path=None,
      sliding_window_length: int = -1,
      sliding_window_context: int = -1,
      custom_split=None,
      which_model=None
  ):

    # Set member variables
    self.sanity_checks = sanity_checks
    self.eval_throttle_secs = eval_throttle_secs
    self.train_tfrecord_path = train_tfrecord_path
    self.dev_tfrecord_path = dev_tfrecord_path
    self.hold_out_fraction = hold_out_fraction
    self.test_tfrecord_path = test_tfrecord_path
    self.model_name = model
    self.label_map_path = label_map_path
    self.model_path = model_path
    self.warm_start_from = warm_start_from
    self.warm_start_vars = warm_start_vars
    self.output_file = output_file
    self.output_confidences = output_confidences
    self.conf_method = conf_method
    self.report_confidences = report_confidences
    self.full_dist = full_dist
    self.bio_dist = bio_dist
    self.logit_dist = logit_dist
    self.temperature = temperature
    self.save_checkpoints_steps = save_checkpoints_steps
    self.keep_checkpoint_hours = keep_checkpoint_hours
    self.train_epochs = train_epochs
    self.max_train_steps = max_train_steps
    self.repeat = repeat
    self.min_epochs = min_epochs
    self.early_stop_patience = early_stop_patience
    self.train_batch_size = train_batch_size
    self.eval_batch_size = eval_batch_size
    self.shuffle_buffer_size = shuffle_buffer_size
    self.random_seed = random_seed
    self.n_gpu = n_gpu
    self.n_cpu = n_cpu
    self.token_calibrate = token_calibrate
    self.logging_names = logging_names
    self.sliding_window_length: int = sliding_window_length
    self.sliding_window_context: int = sliding_window_context
    self.custom_split = custom_split
    self.which_model = which_model

    # if sliding_window_length > 0:
    if self.is_sliding_window:
      if self.sliding_window_context < 0:
        raise ValueError('When sliding_window_length > 0, a non-negative '
                         'sliding_window_context must also be specified!')
      if 2*self.sliding_window_context >= self.sliding_window_length:
        raise ValueError('2*sliding_window_context MUST BE '
                         '< sliding_window_length!')

    # Construct hparams object (self.hparams NOT assigned above)
    self.hparams: HParams = hparams_maybe_override(
        hparams,
        hparams_str=hparams_str,
        hparams_json=hparams_json
    )

    # Create model from Registry
    self.hparams.add_hparam('sliding_window_length',
                            self.sliding_window_length)
    self.hparams.add_hparam('sliding_window_context',
                            self.sliding_window_context)
    self.model = Registries.models[self.model_name](self.hparams)

    # Create data format object (self.data_format NOT assigned above)
    self.data_format: DataFormat = Registries.data_formats[data_format]()

    # Load label_map as raw file and as dict
    with open(self.label_map_path, "rb") as handle:
      self.label_map: Dict[str, int] = pickle.load(handle)
    self.rev_label_map: Dict[int, str] = {v: k
                                          for k, v in self.label_map.items()}

    # Setup logging and perform optional sanity checks
    if self.sanity_checks:
      logging.info("Sanity checking TFRecords...")
      if self.train_tfrecord_path:
        sanity_check_tfrecord(self.train_tfrecord_path, self.data_format,
                              self.hparams, self.label_map)
      if self.dev_tfrecord_path:
        sanity_check_tfrecord(self.dev_tfrecord_path, self.data_format,
                              self.hparams, self.label_map)
      if self.test_tfrecord_path:
        sanity_check_tfrecord(self.test_tfrecord_path, self.data_format,
                              self.hparams, self.label_map)
    else:
      logging.info("WARNING: Skipping sanity checks.")

  @property
  def is_sliding_window(self):
    return self.sliding_window_length > 0

  def _batch_and_prefetch(self, batch_size: int,
                          data: tf.data.Dataset) -> tf.data.Dataset:
    batched_data = data.padded_batch(batch_size,
                                     padded_shapes=self.data_format.shape)
    return batched_data.prefetch(tf.data.experimental.AUTOTUNE)

  def _score(self,
             predictions: Iterable[Dict[str, Any]],
             all_gold_labels: List[List[str]]) -> Tuple[List[str], List[str]]:
    """
    Score predictions against ground truth
    :param predictions: predictions to score
    :param all_gold_labels: ground truth labels
    :return: a 2-tuple of: list of (string) lines to go in predictions.txt,
    list of (string) lines to go in confidences.txt
    """
    predictions_and_labels: List[str] = []
    confidences: List[str] = []
    prediction: Dict[str, Any]
    for sentence_index, prediction in enumerate(predictions):
      if Predictions.SENTENCE_ID.value not in prediction:
        raise ValueError(
            "You're using an old DataFormat; please use Dawn's new one with "
            "SENTENCE_ID!")

      gold_labels: List[str] = all_gold_labels[sentence_index]
      pred_labels: List[str] = self._lookup_pred_labels(prediction)

      if self.is_sliding_window:
        align_len: int = prediction[Features.INPUT_WORD_ALIGNMENT_LENGTH.value]
        alignment: np.ndarray = prediction[Features.INPUT_WORD_ALIGNMENT.value]
        # realign and trim off left and right contexts so we only score
        # the PREDICTION zone at the WORD level
        gold_labels, pred_labels = \
          Trainer._post_process_windowed_predictions(
              gold_labels,
              pred_labels,
              alignment[:align_len],
              self.sliding_window_length,
              self.sliding_window_context
          )
      else:  # no sliding windows
        # truncate any predictions made on the end of sentence padding
        pred_labels = pred_labels[:len(gold_labels)]

      conll_sent_id: int = prediction[Predictions.SENTENCE_ID.value]
      if self.report_confidences:
        output_length = len(gold_labels)

        if 'crf' in self.model_name:
          unary = prediction[Predictions.RAW_TOKEN_SCORES.value][:output_length]
          transition_params = prediction[Predictions.TRANSITION_PARAMS.value]
          scaled_probs = span_confidence(pred_labels, self.label_map,
                                         unary, transition_params)
          scaled_probs = np.exp(scaled_probs)
        else:
          if self.temperature is not None:
            pred_log_probs = \
                prediction[Predictions.RAW_TOKEN_SCORES.value][:output_length]
            pred_log_probs = \
                tf.nn.log_softmax(pred_log_probs / self.temperature)
          else:
            pred_log_probs = \
                prediction[Predictions.TOKEN_SCORES.value][:output_length]
          scaled_probs = np.exp(np.max(pred_log_probs, axis=1))
          scaled_probs = est_span_conf(pred_labels, scaled_probs,
                                       self.conf_method)
        scaled_probs = np.where(  # type: ignore[call-overload]
            scaled_probs > 1.0,
            1.0,  # type: ignore
            scaled_probs)

        if self.full_dist:
          if 'crf' in self.model_name:
            pred_log_probs = token_confidence(unary, transition_params)
          dist_per_token = np.exp(pred_log_probs)
          # create a list of entities stripped of B- or I- designation

          if self.bio_dist:
            labels = [self.rev_label_map[x] for x in range(len(self.label_map))]
            if self.logit_dist:
              pred_logits = \
                  prediction[Predictions.RAW_TOKEN_SCORES.value][:output_length]
              distribution = [
                  [f"{label}:{dist[self.label_map[label]]}" for label in labels]
                  for dist in pred_logits]
            else:
              distribution = [
                  [f"{label}:{dist[self.label_map[label]]}" for label in labels]
                  for dist in dist_per_token]

          else:
            labels = [self.rev_label_map[x][2:] for x in
                      range(len(self.label_map)) if
                      self.rev_label_map[x][:2] == 'B-']

            distribution = [
                [label + ":" + '{:f}'.format(
                    min(1.0,
                        dist[self.label_map["B-" + label]] +
                        dist[self.label_map["I-" + label]]))
                 for label in labels]  # sum the B- and I- tags
                for dist in dist_per_token]  # get dist per token
          for pred, gold, confidence, token_distribution in zip(pred_labels,
                                                                gold_labels,
                                                                scaled_probs,
                                                                distribution):
            # converts distribution from list into csv
            cs_dist = ','.join(token_distribution)
            predictions_and_labels.append(f"{conll_sent_id}\t{gold}\t{pred}\n")
            confidences.append(
                f"{conll_sent_id}\t{gold}\t{pred}\t{confidence}\t{cs_dist}\n")
        else:
          for pred, gold, confidence in zip(pred_labels, gold_labels,
                                            scaled_probs):
            predictions_and_labels.append(f"{conll_sent_id}\t{gold}\t{pred}\n")
            confidences.append(
                f"{conll_sent_id}\t{gold}\t{pred}\t{confidence}\n")
        confidences.append("\n")
      else:
        for pred, gold in zip(pred_labels, gold_labels):
          predictions_and_labels.append(f"{conll_sent_id}\t{gold}\t{pred}\n")

      predictions_and_labels.append("\n")

    return predictions_and_labels, confidences

  def _lookup_pred_labels(self, prediction):
    # TODO this is called with a map of tensors OR a map of ndarrays - STOPIT
    label_ids = prediction[Predictions.TAGS.value]
    if isinstance(label_ids, tf.Tensor):
      label_ids = label_ids.numpy()
    return [self.rev_label_map[idx] for idx in label_ids]

  @staticmethod
  def _post_process_windowed_predictions(
      win_subword_gold_labels: List[str],
      win_subword_pred_labels: List[str],
      win_alignment: np.ndarray,
      win_subword_len: int,
      ctx_subword_len: int,
  ) -> Tuple[List[str], List[str]]:
    """
    Given SUBWORD-level gold and predicted labels from the ENTIRE
    window, returns WORD-level gold and predicted labels from ONLY
    the prediction zone (the central part of the window that excludes the left
    and right context zones).

    :param win_subword_gold_labels: list of gold labels for each subword in
                          the entire window
    :param win_subword_pred_labels: list of predicted labels for each
                          subword in the entire window
    :param win_alignment: a 1D array of the subword indices of the first
                          subword in each word within the window
    :param win_subword_len: the subword length of the entire window (including
                          both contexts and the prediction zone)
    :param ctx_subword_len: the subword length of the two context zones (each
                          one has this length)
    :return: 2-tuple of WORD-level gold labels and predicted labels from
                          ONLY the prediction zone
    """
    assert len(win_subword_gold_labels) == win_subword_len
    assert len(win_subword_pred_labels) == win_subword_len

    # first, collapse subword-level labels back to word-level labels
    # for scoring purposes (doing this BEFORE trimming off the context zones
    # allows the aggregated label applied to a word that starts in the
    # prediction zone to consider its subword labels that may extend into the
    # context zone)
    win_word_gold_labels: List[str] = \
        collapse_subword_values(win_subword_gold_labels,
                                win_alignment, agg_first)
    win_word_pred_labels: List[str] = \
        collapse_subword_values(win_subword_pred_labels,
                                win_alignment, agg_mode_type_first_prefix)

    # next, trim off labels belonging to any words
    # that begin in either of the context zones
    left_ctx_words, right_ctx_words = Trainer._count_context_words(
        win_alignment, win_subword_len, ctx_subword_len)
    right_slice = -right_ctx_words if right_ctx_words > 0 else None

    zone_word_gold_labels = win_word_gold_labels[left_ctx_words:right_slice]
    zone_word_pred_labels = win_word_pred_labels[left_ctx_words:right_slice]

    # check that the number of word-level predictions matches
    # the number of word-level gold labels
    if len(zone_word_pred_labels) != len(zone_word_gold_labels):
      raise ValueError(f"pred length {len(zone_word_pred_labels)} != "
                       f"gold length {len(zone_word_gold_labels)}")

    return zone_word_gold_labels, zone_word_pred_labels

  @staticmethod
  def _count_context_words(win_alignment: Union[Sequence[int], np.ndarray],
                           win_subword_len: int,
                           ctx_subword_len: int) -> Tuple[int, int]:
    """
    Given a word-subword alignment array, counts the number of WORDS (NOT
    subwords) that BEGIN in the left and right context zones of a window.

    :param win_alignment: a 1D array of the subword indices of the first
                          subword in each word within the window
    :param win_subword_len: the subword length of the entire window (including
                          both contexts and the prediction zone)
    :param ctx_subword_len: the subword length of the two context zones (each
                          one has this length)
    :return: 2-tuple of the number of words that begin in the left and right
                          context zones of the window, respectively
    """
    left_context_words = 0
    for i in win_alignment:
      # note that ctx_subword_len == index of 1st subword in prediction zone
      if i < ctx_subword_len:
        left_context_words += 1
      else:
        break

    first_subword_in_right_ctx = win_subword_len - ctx_subword_len
    right_context_words = 0
    for i in reversed(win_alignment):
      if i >= first_subword_in_right_ctx:
        right_context_words += 1
      else:
        break

    return left_context_words, right_context_words

  def _produce_labels(self, fn: Callable[[], tf.data.Dataset]) -> \
      Tuple[List[List[str]], List[np.ndarray]]:
    """ Returns 2-tuple (list of lists of label strings, list of np arrays of
    label indices)
    """
    all_gold_labels: List[List[str]] = []
    all_gold_labels_idx: List[np.ndarray] = []
    for context_map, label_indices in fn():
      n_labels: tf.int64
      label_lengths: np.ndarray = \
        context_map[Features.INPUT_SEQUENCE_LENGTH.value]
      for n_labels, labels in zip(label_lengths, label_indices):
        gold_labels: np.ndarray = labels[:n_labels].numpy()
        all_gold_labels_idx.append(gold_labels)
        all_gold_labels.append([self.rev_label_map[idx]
                                for idx in gold_labels])
    return all_gold_labels, all_gold_labels_idx

  def _new_estimator(self, config=None, warm_start_from=None):
    return tf.estimator.Estimator(
        model_fn=self.model.get_model_fn(),
        model_dir=self.model_path,
        params=self.hparams,
        config=config,
        warm_start_from=warm_start_from
    )

  def _make_serving_input_receiver(self) -> \
      tf.estimator.export.ServingInputReceiver:
    """An input_fn that expects a serialized tf.Example."""
    serialized_tf_example = tf.placeholder(
        dtype=tf.dtypes.string,
        shape=[None],
        name='input_example_tensor')
    receiver_tensors = {'examples': serialized_tf_example}
    parse_fn = self.data_format.parse_example_fn(batch=True)
    # features = parse_fn(serialized_tf_example)
    features, _ = parse_fn(serialized_tf_example)
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

  def _write_best_saved_model(self, estimator: tf.estimator.Estimator):
    """ Writes the given estimator out as the best model
    (in SavedModel format) """
    best_model_dir = str(Path(self.model_path, 'best_model'))
    export_dir = str(Path(self.model_path, 'export'))
    new_best_model_dir = estimator.export_saved_model(
        export_dir,
        serving_input_receiver_fn=self._make_serving_input_receiver,
    )
    if os.path.exists(best_model_dir):
      rmtree(best_model_dir)
    rename(new_best_model_dir, best_model_dir)
    return best_model_dir

  def fit(self) -> None:
    """
    Train the model
    """

    def train_input(take_n: Optional[int] = None,
                    skip_n: Optional[int] = None) -> tf.data.Dataset:
      """ Lambda function for training """
      data = _tfrecords_from_path('train', self.train_tfrecord_path)
      if take_n is not None:
        data = data.take(take_n).concatenate(data.skip(skip_n))
      parsed_data = data.map(self.data_format.parse_single_example_fn)
      shuffled_data = parsed_data.apply(
          tf.data.experimental.shuffle_and_repeat(
              self.shuffle_buffer_size, count=self.repeat,
              seed=self.random_seed))
      return self._batch_and_prefetch(self.train_batch_size, shuffled_data)

    def dev_input(path: str, data_descr: Literal['train', 'dev'],
                  skip_n: Optional[int] = None,
                  take_n: Optional[int] = None) -> tf.data.Dataset:
      """ Returns a dataset for validation """
      data = _tfrecords_from_path(data_descr, path)
      if skip_n is not None:
        data = data.skip(skip_n).take(take_n)
      parsed_data = data.map(self.data_format.parse_single_example_fn)
      return self._batch_and_prefetch(self.eval_batch_size, parsed_data)

    def make_train_and_dev_input_fns(hold_out_fraction: float) -> \
        Tuple[Callable[[], tf.data.Dataset], Callable[[], tf.data.Dataset]]:
      """ Returns a function that will instantiate a validation dataset """
      n_examples_total = number_of_records(self.train_tfrecord_path)

      if self.custom_split is not None:
        custom_split = self.custom_split % 10
        split_count = int(n_examples_total * 0.1)
        split_start = custom_split * split_count
        split_end = split_start + split_count
        return (lambda: train_input(take_n=split_start, skip_n=split_end),
                lambda: dev_input(self.train_tfrecord_path, 'train',
                                  skip_n=split_start, take_n=split_count))

      if hold_out_fraction:
        # Hold out a fraction of train data for dev
        if hold_out_fraction <= 0.0 or hold_out_fraction > 1:
          raise ValueError('hold out fraction must be > 0 <= 1')

        logging.info(f'Holding out {100*hold_out_fraction}% of '
                     f'training data for validation')
        train_fraction = 1. - hold_out_fraction
        n_train = int(n_examples_total * train_fraction)
        logging.info(f'  Using {n_train} of {n_examples_total} '
                     f'training examples for training')
        logging.info(f'  Using remaining {n_examples_total - n_train} '
                     f'training examples for validation.')
        return (lambda: train_input(take_n=n_train, skip_n=-1),
                lambda: dev_input(self.train_tfrecord_path, 'train',
                                  skip_n=n_train, take_n=-1))

      # We're using a separate dev set
      logging.info("Not using hold out fraction")
      n_valid = number_of_records(self.dev_tfrecord_path)
      logging.info(f"  Using all {n_examples_total} examples for training")
      logging.info(f"  Using all {n_valid} examples for validation")
      return (train_input,
              lambda: dev_input(self.dev_tfrecord_path, 'dev'))

    tf.set_random_seed(self.random_seed)
    np.random.seed(self.random_seed)
    config: tf.estimator.RunConfig = get_run_config(
        num_gpu=self.n_gpu,
        num_cpu=self.n_cpu,
        keep_checkpoint_hours=self.keep_checkpoint_hours,
        save_checkpoints_steps=self.save_checkpoints_steps
    )
    pretty_print_hparams(self.hparams)

    ws = tf.estimator.WarmStartSettings(
        ckpt_to_initialize_from=self.warm_start_from,
        vars_to_warm_start=self.warm_start_vars
    ) if self.warm_start_from else None

    # Extra tensors to log during training
    logging_dict = {name: name for name in self.logging_names.split(",")}
    logging_hook = tf.train.LoggingTensorHook(logging_dict, every_n_iter=100)

    estimator = self._new_estimator(config, ws)

    check_tfrecord_input(self.train_tfrecord_path)

    train_input_fn, dev_input_fn = \
      make_train_and_dev_input_fns(self.hold_out_fraction)
    all_gold_labels_val: List[List[str]] = self._produce_labels(dev_input_fn)[0]
    max_f1: float = 0.0
    max_f1_epoch = -1
    patience = self.early_stop_patience
    total_epoch_minutes = 0.0
    epoch = 0
    #n_steps = [128 * 2**i for i in range(10)]
    while True:
      epoch += 1
      logging.info(f'[Epoch {epoch}] Starting training epoch...')
      if ws and epoch == 2:
        estimator = self._new_estimator(config, None)  # stop warm-starting
        logging.info('  Disabled warm-start')
      epoch_start_time = time.time()
      estimator.train(input_fn=train_input_fn, hooks=[logging_hook])
      metrics: Dict[str, tf.Tensor] = estimator.evaluate(input_fn=dev_input_fn)

      logging.info('  Evaluation metrics:')
      for k, v in metrics.items():
        logging.info(f'    {k}: {v}')

      # TODO see if we can compute F1 within call to estimator.evaluate above
      data_type = 'held-out train' if self.hold_out_fraction else 'dev'
      logging.info(f'    Making predictions on {data_type} data...')
      val_score_lst: List[str] = self._score(
          estimator.predict(dev_input_fn),
          all_gold_labels_val,
      )[0]
      f1: float = 100 * full_metrics(evaluate(val_score_lst)).overall.fscore
      logging.info(f'    F1: {f1:.2f}%')

      epoch_minutes = (time.time() - epoch_start_time) / 60
      logging.info(f'    epoch time: {epoch_minutes:.2f} minutes')
      total_epoch_minutes += epoch_minutes

      epoch_dir = str(Path(self.model_path, f"epoch_{epoch}_model"))
      export_dir = str(Path(self.model_path, 'export'))
      temp_save_dir = estimator.export_saved_model(
          export_dir,
          serving_input_receiver_fn=self._make_serving_input_receiver,
      )
      if os.path.exists(epoch_dir):
        rmtree(epoch_dir)
      rename(temp_save_dir, epoch_dir)

      is_improved = f1 > max_f1
      if is_improved:
        max_f1 = f1
        max_f1_epoch = epoch
        best_model_dir = self._write_best_saved_model(estimator)
        logging.info(f'  Wrote new best model (F1={f1:.2f}%) '
                     f'to "{best_model_dir}".')
        patience = self.early_stop_patience  # since we improved, reset patience

      if epoch == self.train_epochs:
        logging.info(f'Completed requested {self.train_epochs} epochs.')
        break
      if epoch < self.min_epochs:  # should min_epochs trump train_epochs?
        logging.info(f'  Epoch {epoch} < minimum epochs ({self.min_epochs})')
      elif not is_improved:
        patience -= 1
        logging.info(
            f'  Losing patience ({patience}/{self.early_stop_patience}).')
        if not patience and epoch + 1 < self.train_epochs:
          logging.info(f'*** Early stopping after {epoch} epochs ***\n'
                       f'  No F1 improvement during '
                       f'the last {self.early_stop_patience} epochs.')
          break

    logging.info('')
    logging.info(f'Trained {epoch} epochs '
                 f'in ~{total_epoch_minutes:.2f} minutes')
    logging.info(f'Avg time/epoch: {total_epoch_minutes/epoch:.2f} minutes')
    logging.info(f'Max validation F1 (in epoch {max_f1_epoch}): {max_f1:.2f}%')

  def predict(self) -> None:
    """
    Evaluate the model
    """

    def test_input() -> tf.data.Dataset:
      """ Lambda function for testing """
      data = _tfrecords_from_path('test', self.test_tfrecord_path)
      return data.batch(self.eval_batch_size)

    tf.set_random_seed(self.random_seed)
    np.random.seed(self.random_seed)

    check_tfrecord_input(self.test_tfrecord_path)

    best_model_dir = str(Path(self.model_path, 'best_model'))
    if self.which_model:
      best_model_dir = str(Path(self.model_path,
                                f"epoch_{self.which_model}_model"))
    imported = tf2.saved_model.load(  # type: ignore[attr-defined]
        best_model_dir, tags=['serve'])
    if logging.level_debug():
      sig_names_str = '\n\t'.join(list(imported.signatures.keys()))
      logging.debug(f'\nImported signatures:\n\t{sig_names_str}')
    predict_fn = imported.signatures['serving_default']

    label_map_str = '\n\t'.join({f'{k} {v}'
                                 for k, v in self.rev_label_map.items()})
    logging.info(f'Label map:\n\t{label_map_str}')

    # Get test labels
    all_gold_labels_test: List[List[str]]
    all_gold_labels_idx: List[np.ndarray]
    all_gold_labels_test, all_gold_labels_idx = self._produce_labels(
        lambda: test_input().map(
            self.data_format.parse_example_fn(batch=True))
    )

    # Get predictions
    predictions: List[Dict[str, tf.Tensor]] = list(
        # TODO can we eliminate 'unbatch'?
        test_input().map(predict_fn).unbatch())
    test_score_lst, confidences = self._score(predictions, all_gold_labels_test)

    report(evaluate(test_score_lst))

    if self.token_calibrate:

      # grab the parent directory and pass it to token_calibration_metrics
      parent_folder = Path(self.output_file).parent

      from griffin.token_calibration import token_calibration_metrics
      is_crf = 'crf' in self.model_name
      calibrated, mean_predicted_values, correct_per_bin, calib, gold_labels = \
        token_calibration_metrics(predictions,
                                  all_gold_labels_idx,
                                  self.hparams.output_dim,  # type: ignore
                                  self.model_name,
                                  parent_folder,
                                  is_crf)
      logging.info("Token Calibration")
      logging.info(f"  Calibrated: {calibrated}")
      logging.info(f"  calib: {calib}")
      if calibrated:
        for correct_in_bin, mean_predicted_value, gold_label in zip(
            correct_per_bin, mean_predicted_values, gold_labels):
          logging.info(
              f"mean predicted value: {mean_predicted_value} "
              f"Calibration error: {correct_in_bin:.6f} "
              f"Gold Label: {gold_label} ")

    # Write predictions to file
    if self.report_confidences:
      with open(self.output_file, 'w') as out_file:
        with open(self.output_confidences, 'w') as conf_file:
          for line, confidence in zip(test_score_lst, confidences):
            out_file.write(line)
            conf_file.write(confidence)
    else:
      with open(self.output_file, 'w') as out_file:
        for line in test_score_lst:
          out_file.write(line)
