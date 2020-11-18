#! /usr/bin/env python

# Copyright 2019 Johns Hopkins University. All Rights Reserved.
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

# pylint: disable=import-outside-toplevel

"""Given a trained NER model, try to calibrate the confidence it
assigns to its predictions at the token level.
"""

import time
from pathlib import Path
from typing import Dict
from typing import List

import numpy as np
import tensorflow as tf
from absl import logging

from ner.decode import crf_get_token_probs
from ner.features import Predictions


# vectorized, 6 time speedup
def get_one_hot(tensor, output_dim):
  """ tensor: of shape (sentence_length, )
  returns (sentence_length, output_dim) matrix, one_hot version of matrix """
  if output_dim == 1:
    return [x if x == 0 else 1 for x in tensor]
  return np.eye(output_dim)[tensor]


# for loop
def get_one_hot_for_loop(gold_labels_at_idx, output_dim):
  """ returns (sentence_length, output_dim matrix """
  all_one_hot_gold = []
  for pos in gold_labels_at_idx:
    one_hot_gold = np.zeros(output_dim, dtype=tf.int32)
    one_hot_gold[pos] = 1
    all_one_hot_gold.append(one_hot_gold[1:])
  return all_one_hot_gold


def bin_by_num(distribute, n_bins):
  """ Distribute distribute into n_bins """
  bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
  binidx = np.digitize(distribute, bins) - 1
  return np.bincount(binidx, minlength=len(bins))


def bin_by_bins(distribute, bins):
  """ Distribute distribute into an array of bins, increasing
Assumes additional left-bounded bin w/ elem = 1 """
  binidx = np.digitize(distribute, bins) - 1
  return np.bincount(binidx, minlength=len(bins))[:-1]


def uniform_bin(n_bins):
  """ Creates left-bounded bins w/ n_bins + 1 elemes, last one = 1 """
  return np.linspace(0., 1. + 1e-8, n_bins + 1)


def get_token_probs(predictions, gold_labels_per_sentence, output_dim):
  """ returns:
  class_probs of (batch_size * sentence_length * output_dim, )
  correct_probs of at most (batch_size * sentence_length * output_dim, )
  gold_labels, one hot vectors of (batch_size * sentence_length * output_dim, )
  """

  class_probs = np.array([])
  gold_labels = np.array([])  # for sklearn, but as of now not really needed
  for sentence_index, prediction in enumerate(predictions):
    sentence_length = prediction[Predictions.LENGTH.value]

    pred_log_probs = \
      prediction[Predictions.TOKEN_SCORES.value][:sentence_length]
    gold_labels_at_idx = \
      gold_labels_per_sentence[sentence_index][:sentence_length]
    pred, gold = get_token_probs_one_sent(pred_log_probs,
                                          gold_labels_at_idx,
                                          output_dim)
    class_probs = np.append(class_probs, pred)
    gold_labels = np.append(gold_labels, gold)

  return class_probs, gold_labels


def get_token_probs_one_sent(pred_log_probs, gold_labels_at_idx, output_dim):
  """
  Given pred_log_probs (seq_len x num_tags)
  Project gold_labels_at_idx into one hot
  """
  # this is pred_log_probs
  prob_per_token_at_idx = np.exp(pred_log_probs)
  # collapse probability if binary
  if output_dim == 1:
    if prob_per_token_at_idx.ndim > 1:
      prob_per_token_at_idx = 1 - prob_per_token_at_idx
    else:
      prob_per_token_at_idx = np.log(prob_per_token_at_idx)

  # (sentence_length, ), elem = correct tag at position
  all_one_hot_gold = get_one_hot(gold_labels_at_idx, output_dim)
  return prob_per_token_at_idx, all_one_hot_gold


def get_crf_token_probs(predictions, gold_labels_per_sentence, output_dim):
  """ returns:
  class_probs of (batch_size * sentence_length * output_dim, )
  correct_probs of at most (batch_size * sentence_length * output_dim, )
  gold_labels, one hot vectors of (batch_size * sentence_length * output_dim, )
  """

  class_probs = np.array([])
  gold_labels = np.array([])  # for sklearn, but as of now not really needed
  for sentence_index, prediction in enumerate(predictions):
    sentence_length = prediction[Predictions.LENGTH.value]
    # (sentence_length x output_dim), ignore non-entity tag for now
    unary = prediction[Predictions.RAW_TOKEN_SCORES.value][:sentence_length]
    transition_params = prediction[Predictions.TRANSITION_PARAMS.value]

    distribution = crf_get_token_probs(sentence_length, unary,
                                       transition_params)
    class_probs = np.append(class_probs, distribution)
    # (sentence_length, ), elem = correct tag at position
    gold_labels_at_idx = gold_labels_per_sentence[sentence_index][
        :sentence_length]
    # now (sentence_length, output_dim)
    all_one_hot_gold = get_one_hot(gold_labels_at_idx, output_dim)
    gold_labels = np.append(gold_labels, all_one_hot_gold)
  return class_probs, gold_labels


def get_calib_error(class_probs, gold_labels, base_bin, mean_predicted_value):
  """ Absolute value of average by correct per bin, weighed by all
  probabilities """

  # midpt of our bins
  mean_predicted_value = average_bin(base_bin)

  # given token probabilities and gold labels, produce probabilities assigned
  # to labels
  correct_probs = class_probs[np.nonzero(gold_labels)]
  binned_correct = bin_by_bins(correct_probs, base_bin)
  binned_class = bin_by_bins(class_probs, base_bin)

  # ignore divide by 0 or NaN warnings
  with np.errstate(divide='ignore', invalid='ignore'):
    correct_per_bin = binned_correct / binned_class
  # correct NaN to 0
  correct_per_bin[np.isnan(correct_per_bin)] = 0

  calib = np.around(
      np.dot(np.abs(mean_predicted_value[:-1] - correct_per_bin[:-1]),
             binned_class[:-1]) / np.sum(binned_class[:-1]),
      5)
  return binned_correct, binned_class, correct_per_bin, calib


def adaptive_binning(class_probs, gold_labels, model_name, output_path):
  """ Does adaptive binning, presumably... """
  try:
    from sklearn.calibration import calibration_curve
  except ImportError:
    logging.error("Sklearn is not installed")
    return (False, None, None, None)
  correct_per_bin, mean_predicted_value = calibration_curve(gold_labels,
                                                            class_probs,
                                                            n_bins=500,
                                                            strategy='quantile')

  calib = np.sqrt(np.sum(
      np.square(mean_predicted_value - correct_per_bin)) / correct_per_bin.size)
  mean_predicted_value = np.concatenate(
      [mean_predicted_value, np.array([mean_predicted_value[-1]])])
  reliability_plot(correct_per_bin, mean_predicted_value, calib, model_name,
                   output_path)
  return (True, mean_predicted_value, correct_per_bin, calib)


def get_sharpness(fraction_of_positives, binned_correct, binned_class):
  # correct_per_bin - reduced_correct_per_bin ... binned_class
  pct_predicted = np.sum(binned_correct) / np.sum(binned_class)
  sharpness = np.around(np.dot((fraction_of_positives - pct_predicted) ** 2,
                               binned_class) / np.sum(binned_class), 5)
  perfect_sharpness = (pct_predicted) * (1 - pct_predicted) ** 2 + (
      1 - pct_predicted) * (pct_predicted) ** 2
  return pct_predicted, sharpness, perfect_sharpness


def reliability_plot(correct_per_bin, mean_predicted_value, calibration,
                     model_name, output_path):
  """ Generate reliability diagram for given model """
  try:
    import matplotlib as matplot
    matplot.use('Agg')
    import matplotlib.pyplot as plt
    # reliability_plot(correct_per_bin, base_bin, calib_error, sharpness,
    # model_name)
  except ImportError:
    logging.error("Install Matplotlib to generate reliability diagram.")
    return 0
  fig = plt.figure()
  ax = plt.subplot(111)
  ax.plot(mean_predicted_value[:-1], correct_per_bin, color="g")
  ax.plot([0, 1], [0, 1], color="k", linestyle=":")
  plt.xlabel(f"mean predicted value\n\nCalibration error: {calibration:.6f}")
  plt.ylabel("fraction of positives")
  plt.title(f"Reliability plot for model: {model_name}")
  plt.tight_layout()
  timestr = time.strftime("%Y%m%d-%H%M%S")
  filename = "reliability_plot_" + model_name + "_" + timestr + ".png"
  file_to_write = output_path / Path(filename)
  logging.info(f"Reliability plot file: {file_to_write}")
  fig.savefig(file_to_write, bbox_inches='tight')

  return 1


def average_bin(base_bin):
  # shallow copy
  def average(a, b):
    """ Bins indicate left boundaries, avg function to get midpt of bins """
    return (a + b) / 2

  return [average(a, b) for a, b in zip(base_bin[:-1], base_bin[1:])]


def token_calibration_metrics(predictions: List[Dict[str, tf.Tensor]],
                              gold_labels_per_sentence,
                              output_dim, model_name, output_path, is_crf):
  """ Obtains tokens, calculates error and sharpness, plot """
  # for now, hard-code the base_bin aka mean_predicted_value

  if is_crf:
    class_probs, gold_labels = get_crf_token_probs(predictions,
                                                   gold_labels_per_sentence,
                                                   output_dim)
  else:
    class_probs, gold_labels = get_token_probs(predictions,
                                               gold_labels_per_sentence,
                                               output_dim)
  (calibrated, mean_predicted_value, correct_per_bin, calib) = adaptive_binning(
      class_probs, gold_labels, model_name, output_path)

  return calibrated, mean_predicted_value, correct_per_bin, calib, gold_labels
