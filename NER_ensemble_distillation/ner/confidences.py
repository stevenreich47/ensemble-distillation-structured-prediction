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
# ==============================================================================

"""
TODO describe this module
"""

import numpy as np


def token_tuples(pred_labels):
  tuples = []
  for idx, _ in enumerate(pred_labels):
    tuples.append((idx, idx + 1))
  return tuples


def span_tuples(pred_labels):
  """ Return the start and end indices corresponding to each
  segment of `pred_labels`, including `O` as a special case
  which always has length one.

  """

  tuples = []
  in_span = False
  begin = 0
  end = 0
  for idx, elem in enumerate(pred_labels):
    if not in_span and elem[0] == 'B':
      # start of a new span
      in_span = True
      begin = idx
    elif in_span and elem[0] == 'B':
      # terminate prev span
      end = idx
      tuples.append((begin, end))
      # and start a new one
      begin = idx
    elif in_span and elem[0] == 'O':
      # terminate prev span
      in_span = False
      end = idx
      tuples.append((begin, end))
      # and then also the '0' tag
      tuples.append((end, end + 1))
    elif not in_span and elem[0] == 'O':
      tuples.append((idx, idx + 1))
  # handle cases where sentence ends, but not on '0'
  if in_span:
    end = len(pred_labels)
    tuples.append((begin, end))
  return tuples


def est_span_conf(pred_labels, scaled_probs, method='average'):
  """ Modifies token probabilities to account for entity spans """
  tuples = span_tuples(pred_labels)
  scaled_probs = scaled_probs.astype(np.float32)

  # max, min, average
  for tup in tuples:
    if method == 'average':
      scaled_probs[tup[0]:tup[1]] = np.mean(scaled_probs[tup[0]:tup[1]])
    elif method == 'max':
      scaled_probs[tup[0]:tup[1]] = np.max(scaled_probs[tup[0]:tup[1]])
    elif method == 'min':
      scaled_probs[tup[0]:tup[1]] = np.min(scaled_probs[tup[0]:tup[1]])
  return scaled_probs
