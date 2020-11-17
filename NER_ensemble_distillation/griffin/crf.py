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
# ============================================================================

# pylint: disable=invalid-name

""" Functions to compute confidences for a CRF """

import tensorflow as tf
import numpy as np

from griffin.confidences import span_tuples


def forward(inputs, state, transition_params, sequence_lengths,
            state_mask=None, back_prop=False, parallel_iterations=10,
            swap_memory=True):
  """Computes the alpha values in a linear-chain CRF.

    See http://www.cs.columbia.edu/~mcollins/fb.pdf for reference.

    The current masking approach is not numerically stable when
    computing gradients. Therefore, this should only be used at
    inference time with `back_prop=False` and not when training a
    model. See:

    https://github.com/tensorflow/tensorflow/issues/11756

    Args:
      inputs: A [batch_size, max_seq_len, num_tags] matrix of unary potentials.
      state: A [batch_size, num_tags] matrix containing the previous alpha
         values.
      transition_params: A [num_tags, num_tags] matrix of binary potentials.
          This matrix is expanded into a [1, num_tags, num_tags] in preparation
          for the broadcast summation occurring within the cell.
      sequence_lengths: A [batch_size] vector of true sequence lengths.
      state_mask: A [batch_size, max_seq_len, num_tags] boolean tensor.
      backprop: Enable support for gradient calculation.
      parallel_iterations: Number of iterations to perform in parallel.
      swap_memory: CPU-GPU memory swapping.

    Returns:
      new_alphas: A [batch_size, num_tags] matrix containing the
          new alpha values.

  """

  sequence_lengths = tf.cast(sequence_lengths, dtype=tf.int32)
  last_index = tf.maximum(
      tf.constant(0, dtype=sequence_lengths.dtype), sequence_lengths - 1)
  inputs = tf.transpose(inputs, [1, 0, 2])
  transition_params = tf.expand_dims(transition_params, 0)

  if state_mask is None:
    state_mask = tf.ones_like(inputs)
  else:
    state_mask = tf.transpose(state_mask, [1, 0, 2])

  def _scan_fn(_state, _inputs_and_mask):
    _inputs, _mask = _inputs_and_mask
    _state = tf.expand_dims(_state, 2)
    transition_scores = tf.add(_state, transition_params,
                               name='transition_scores')
    masked_transition_scores = tf.multiply(transition_scores,
                                           tf.expand_dims(_mask, 1))
    final_scores = tf.where(masked_transition_scores >= np.inf,
                            -np.inf, masked_transition_scores)
    new_alphas = tf.add(_inputs, tf.reduce_logsumexp(final_scores, [1]))
    return new_alphas

  all_alphas = tf.transpose(tf.scan(_scan_fn, (inputs, state_mask), state,
                                    parallel_iterations=parallel_iterations,
                                    back_prop=back_prop,
                                    swap_memory=swap_memory),
                            [1, 0, 2])
  # add first state for sequences of length 1
  all_alphas = tf.concat([tf.expand_dims(state, 1), all_alphas], 1)

  idxs = tf.stack([tf.range(tf.shape(last_index)[0]), last_index], axis=1)
  return tf.gather_nd(all_alphas, idxs)


def log_norm(inputs, sequence_lengths, transition_params,
             state_mask=None):
  """Computes the normalization for a CRF.

    Args:
      inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
          to use as input to the CRF layer.
      sequence_lengths: A [batch_size] vector of true sequence lengths.
      transition_params: A [num_tags, num_tags] transition matrix.
      state_mask: A [batch_size, max_seq_len, num_tags] boolean mask.

    Returns:
      log_norm: A [batch_size] vector of normalizers for a CRF.

  """
  sequence_lengths = tf.cast(sequence_lengths, dtype=tf.int32)

  if len(inputs.shape) != 3:
    raise ValueError(
        (f"Expected `inputs` to have shape"
         f" [batch_size, max_seq_len, num_states]"
         f" but `inputs` has shape {inputs.shape}"))

  # Split up the first and rest of the inputs in preparation for the forward
  # algorithm.
  first_input = tf.slice(inputs, [0, 0, 0], [-1, 1, -1])
  first_input = tf.squeeze(first_input, [1])

  if state_mask is None:
    state_mask = tf.ones_like(inputs)

  # If max_seq_len is 1, we skip the algorithm and simply reduce_logsumexp
  # over the "initial state" (the unary potentials).
  def _single_seq_fn():
    masked_first_input = tf.multiply(first_input, state_mask[:, 0, :])
    masked_first_input = tf.where(masked_first_input >= np.inf,
                                  -np.inf, masked_first_input)
    _log_norm = tf.reduce_logsumexp(masked_first_input, [1])
    # Mask `log_norm` of the sequences with length <= zero.
    _log_norm = tf.where(
        tf.less_equal(sequence_lengths, 0), tf.zeros_like(_log_norm),
        _log_norm)
    return _log_norm

  def _multi_seq_fn():
    """Forward computation of alpha values."""
    rest_of_input = tf.slice(inputs, [0, 1, 0], [-1, -1, -1])
    rest_of_mask = tf.slice(state_mask, [0, 1, 0], [-1, -1, -1])

    # Compute the alpha values in the forward algorithm in order to get the
    # partition function.
    initial_alphas = first_input * state_mask[:, 0, :]
    alphas = forward(rest_of_input, initial_alphas,
                     transition_params,
                     sequence_lengths, rest_of_mask)
    _log_norm = tf.reduce_logsumexp(alphas, [1])
    # Mask `log_norm` of the sequences with length <= zero.
    _log_norm = tf.where(
        tf.less_equal(sequence_lengths, 0), tf.zeros_like(_log_norm),
        _log_norm)
    return _log_norm

  return tf.cond(
      tf.equal(tf.shape(inputs)[1], 1), _single_seq_fn, _multi_seq_fn)


def token_confidence(scores, transition_params):
  """Expects inputs to be for a single example, and we perform T x K
  forward passes in parallel where T is the sequence length and K is
  the number of states.

  NOTE: The first dimension of `scores` should correspond to the
  length of the input sequence; in particular, `scores` should not be
  padded.

  TODO: Make this work in Graph mode so that it can be executed
  directly from a `SavedModel`.

  TODO: By construction, the sequence lengths are identical, so it
  should be possible to perform static optimizations such as
  "unrolling" the scan loop in `forward` to improve inference speed.

  """
  assert tf.executing_eagerly(), "Expected Eager execution"

  if len(scores.shape) != 2:
    raise ValueError("Expected shape [sequence_length, num_states]")

  T = scores.shape[0]  # pylint: disable=C0103
  K = scores.shape[1]  # pylint: disable=C0103
  B = T * K + 1
  masks = np.ones((B, T, K), dtype=np.float32)
  batch_index = 1
  for t in range(T):
    for k in range(K):
      masks[batch_index][t][:] = -np.inf
      masks[batch_index][t][k] = 1.
      batch_index += 1
  lens = np.array([T] * B, dtype=np.int32)
  scores = np.tile(np.expand_dims(scores, 0), (B, 1, 1))
  result = log_norm(scores, lens, transition_params, masks).numpy()
  z = result[0]
  return np.reshape(result[1:], [T, K]) - z


def span_confidence(labels, label_map, scores, transition_params):
  """Expects inputs to be for a single example, and batching happens at
  the level of the sequence length. This may cause problems for very
  long sequences when running on the GPU. However, because (1) we
  already have the scores fixed at this point and (2) we are not
  computing gradients, there should be fairly minimal memory overhead,
  and so the chance of hitting an out-of-memory error should be small
  even when running on very long sequences. This risk may eliminated
  when using sliding windows for inference, where the sequence length
  is controlled.

  TODO: Vectorize this so that in can be run directly from a
  `SavedModel` rather than post-hoc given predictions from a model.
  Currently there is unnecessary overhead in passing things from
  Python to TensorFlow, which may cause this function to run faster on
  CPU as a result of the latency associated with host <-> device
  transfer. The main challenge is that we currently rely on imperative
  logic to identify the span boundaries.

  TODO: By construction the sequence lengths are identical, so it
  should be possible to perform some static optimizations such as
  "unrolling" the scan loop in `forward`.

  """
  if len(scores.shape) != 2:
    raise ValueError("Expected shape [sequence_length, num_states]")

  assert len(labels) == scores.shape[0], f"{len(labels)} != {scores.shape[0]}"
  assert tf.executing_eagerly(), "Expected Eager execution"
  label_indices = [label_map[label] for label in labels]
  n_states = scores.shape[1]
  seq_len = scores.shape[0]
  spans = span_tuples(labels)
  n_spans = len(spans)
  assert n_spans
  batch_size = n_spans + 1
  log_probs = np.zeros(shape=(seq_len), dtype=np.float32)
  batch_state_mask = np.ones((batch_size, seq_len, n_states),
                             dtype=np.float32)
  for i in range(batch_size):  # O(T)
    if i < 1:
      pass  # unconstrained to compute Z(x)
    else:
      span_tup = spans[i - 1]
      for j in range(span_tup[0], span_tup[1]):
        label_index = label_indices[j]
        batch_state_mask[i, j, :] = -np.inf
        batch_state_mask[i, j, label_index] = 1.
  batch_state_mask = tf.constant(batch_state_mask)
  batch_scores = tf.tile(tf.expand_dims(scores, 0),
                         [batch_size, 1, 1])
  batch_lengths = tf.tile(
      tf.expand_dims(
          tf.constant(np.array(seq_len, dtype=np.int32)), 0),
      [batch_size])
  assert len(batch_scores.shape) == 3
  assert len(transition_params.shape) == 2
  assert len(batch_state_mask.shape) == 3
  assert batch_scores.shape == batch_state_mask.shape
  z_vals = log_norm(batch_scores, batch_lengths, transition_params,
                    state_mask=batch_state_mask)
  z_vals = z_vals.numpy()  # potential device -> host
  z = z_vals[0]
  z_index = 1
  for i in range(n_spans):  # O(T)
    tup = spans[i]
    z_theta = z_vals[z_index]
    log_probs[tup[0]:tup[1]] = z_theta - z
    z_index += 1
  return log_probs
