# Copyright 2019 Johns Hopkins University
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
# ============================================================================

""" Functions that handle making predictions given a posterior distribution """
from typing import List
from typing import Tuple

import numpy as np
from scipy.special import logsumexp

from griffin.confidences import span_tuples


def viterbi_decode(score: np.ndarray,
                   transition_params: np.ndarray = None,
                   transition_weights: np.ndarray = None,
                   ) -> Tuple[List[int], float]:
  """Decode the highest scoring sequence of tags outside of TensorFlow.
  Note that unlike beam search, the Viterbi algorithm makes a Markov
  assumption concerning state transitions. This allows us to
  consider all possible sequences efficiently using dynamic
  programming but is only appropriate for models like HMMs and CRFs.

  Args:
    score: A [seq_len, num_tags] matrix of unary potentials.
    transition_params: A [num_tags, num_tags] matrix of binary potentials.
    transition_weights: A [seq_len, num_tags, num_tags] tensor of state
      transition weights. The transition params are multiplied by these weights.
      To disable a transition from state i at previous timestep to state j at
      current timestep, use negative infinity:
      `transition_weights[i][j] = -np.inf`.

  Returns:
    viterbi: A [seq_len] list of integers containing the highest scoring tag
      indices.
    viterbi_score: A float containing the score for the Viterbi sequence.

  """

  n_tags = score.shape[-1]
  seq_len = score.shape[0]
  if transition_params is None:
    transition_params = np.zeros((n_tags, n_tags), dtype=np.float32)

  if transition_weights is None:
    transition_weights = np.ones((seq_len, n_tags, n_tags), dtype=np.float32)

  score = np.concatenate([np.zeros([1, score.shape[1]]), score])
  transition_weights = np.concatenate(
      [np.ones([1, n_tags, n_tags]), transition_weights])

  trellis = np.zeros_like(score)
  backpointers = np.zeros_like(score, dtype=np.int32)
  trellis[0] = score[0]

  # loop t over rest of sequence length
  for t in range(1, score.shape[0]):
    # prev_alpha to this timestep, transition from prev_state to now +
    # prev_alpha + transition + emission
    v = np.expand_dims(trellis[t - 1], 1) + transition_params * \
        transition_weights[t] + score[t]
    trellis[t] = np.max(v, 0)
    backpointers[t] = np.argmax(v, 0)

  viterbi = [np.argmax(trellis[-1])]

  for bp in reversed(backpointers[1:]):
    viterbi.append(bp[viterbi[-1]])
  viterbi.reverse()

  viterbi_score = np.max(trellis[-1])
  return viterbi[1:], viterbi_score - 1


def fwd(score, transition_params=None, transition_weights=None,
        return_trellis=False):
  """Calculates the partition function or normalization factor with the
  forward algorithm. In order to constrain the initial tag,
  we expand the seq_len by 1

  Args:
    score: A [seq_len, num_tags] matrix of unary potentials.
    transition_params: A [num_tags, num_tags] matrix of binary potentials.
    transition_weights: A [seq_len, num_tags, num_tags] tensor of state
      transition weights. The transition params are multiplied by these weights.
      To disable a transition from state i to state j, use negative infinity:
      `transition_weights[i][j] = -np.inf`.

  Returns:
    end_scores: partition score for the inputs
  """
  n_tags = score.shape[-1]
  seq_len = score.shape[0]
  if transition_params is None:
    transition_params = np.zeros([n_tags, n_tags], dtype=np.float32)

  if transition_weights is None:
    transition_weights = np.ones((seq_len, n_tags, n_tags), dtype=np.float32)

  # stack an additional dummy layer
  score = np.concatenate([np.zeros([1, score.shape[1]]), score])
  # we can go from initial padding to anywhere
  transition_weights = np.concatenate(
      [np.ones([1, n_tags, n_tags]), transition_weights])

  # the rest follows unconstrained fwd bwd
  trellis = np.zeros_like(score, dtype=np.float32)
  trellis[0] = score[0]
  for t in range(1, score.shape[0]):
    # prev_alpha + transition + emission
    masked_transitions = transition_params * transition_weights[t]
    # two negatives make a positive, so flip it back
    masked_transitions[masked_transitions >= np.inf] = -np.inf

    v = np.expand_dims(trellis[t - 1], 1) + masked_transitions + score[t]
    trellis[t] = logsumexp(v, axis=0)
  end_scores = logsumexp(trellis[-1])

  if return_trellis:
    return end_scores, trellis

  return end_scores


def crf_get_token_probs(output_length, unary, transition_params=None,
                        transition_weights=None):
  """Get the probabilities for transitions between token tags"""
  n_tags = unary.shape[-1]
  seq_len = unary.shape[0]

  if transition_weights is None:
    original_transition_weights = np.ones((seq_len, n_tags, n_tags),
                                          dtype=np.float)
  else:
    original_transition_weights = np.empty_like(transition_weights)
    original_transition_weights[:] = transition_weights

  z = fwd(unary, transition_params, original_transition_weights)
  labels = np.arange(n_tags)
  all_dist = np.zeros([seq_len, n_tags])
  for idx in range(output_length):
    dist = []
    for label in labels:
      transition_weights = np.empty_like(original_transition_weights)
      transition_weights[:] = original_transition_weights
      col_targets = np.delete(labels, label)
      transition_weights[idx][:, col_targets] = -np.inf
      z_theta = fwd(unary, transition_params, transition_weights)
      prob = np.exp(z_theta - z)
      dist.append(prob)

    # sanity check by running viterbi
    all_dist[idx] = dist
  return all_dist


def crf_get_conf(pred_labels, scaled_probs, unary, label_map,
                 transition_params=None, transition_weights=None):
  """Calculates the confidence estimate for each entity span.

  Args:
    pred_labels: A [seq_len] list of B-, I-, and O tags
    unary: A [seq_len, num_tags] matrix of unary potentials.
    label_map: So we can lookup the numeric values of the labels
    transition_params: A [num_tags, num_tags] matrix of binary potentials.
    transition_weights: A [seq_len, num_tags, num_tags] tensor of state
      transition weights. The transition params are multiplied by these weights.
      To disable a transition from state i to state j, use negative infinity:
      `transition_weights[i][j] = -np.inf`.

  Returns:
    end_scores: partition score for the inputs
  """
  n_tags = unary.shape[-1]
  seq_len = unary.shape[0]

  if transition_weights is None:
    original_transition_weights = np.ones((seq_len, n_tags, n_tags),
                                          dtype=np.float32)
  else:
    original_transition_weights = np.empty_like(transition_weights)
    original_transition_weights[:] = transition_weights
  # don't need to initialize transition_params: will be handled in call to fwd
  tuples = span_tuples(pred_labels)
  # calculate partition without constraints
  z = fwd(unary, transition_params, original_transition_weights)
  labels = np.arange(n_tags)
  # max, min, average
  scaled_probs = scaled_probs.astype(np.float32)

  # print('='*80, "begin")
  for tup in tuples:  # get idx of entity span
    transition_weights = np.empty_like(original_transition_weights)
    transition_weights[:] = original_transition_weights
    for idx in range(tup[0], tup[1]):  # get spec. tag @ idx
      curr = pred_labels[idx]
      curr_tag_idx = label_map[curr]
      if idx == tup[0]:
        # if first in seq, 0 out everything that doesn't lead to tag_idx
        col_targets = np.delete(labels, curr_tag_idx)
        transition_weights[idx][:, col_targets] = -np.inf
        # any prev state can only lead to curr_tag_idx
      else:
        # select all prev_tags that != prev_tag_idx
        row_targets = np.delete(labels, prev_tag_idx)
        # and constrain the possibilities
        transition_weights[idx][row_targets, :] = -np.inf
        # this leaves just the prev_tag_idx row unaffected
        # now constrain all transitions but curr_tag_idx
        col_targets = np.delete(labels, curr_tag_idx)
        transition_weights[idx][:, col_targets] = -np.inf
        # use the prev as a constraint to lead to curr
      prev_tag_idx = curr_tag_idx
    # after we have generated our transition_weights
    z_theta = fwd(unary, transition_params, transition_weights)
    # sanity check by running viterbi
    final_prob = np.exp(z_theta - z)
    scaled_probs[tup[0]:tup[1]] = final_prob
  return scaled_probs
