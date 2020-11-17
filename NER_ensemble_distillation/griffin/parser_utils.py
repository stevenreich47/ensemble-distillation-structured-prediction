# Copyright 2019 Johns Hopkins University.
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

""" Utilities common to `Parser` classes. """
from collections import Counter
from typing import List, Dict, TypeVar, Callable
from typing import Sequence
from typing import Union

import numpy as np

from griffin.conll import get_tag_type, is_begin_tag


def tags_to_ids(tags_as_strings: List[str],
                label_map: Dict[str, int]) -> List[int]:
  labels: List[int] = []
  for tag_str in tags_as_strings:
    tag_idx = label_map[tag_str]
    labels.append(tag_idx)
  return labels


def tags_to_classify_ids(tags_as_strings, label_map):
  """The label map should contain a mapping from entity **types** to
  indices. This returns, for each entity span in `tags_as_strings`,
  a list of the corresponding types.

  """
  ret = []
  for idx, tag in enumerate(tags_as_strings):
    if is_begin_tag(tag):
      etype = get_tag_type(tag)
      etype_id = label_map[etype]
      ret.append((idx, etype_id))
  return ret


T = TypeVar('T')


def agg_concat_bert(subword_vals: List[str]) -> str:
  """Concatenates bert subwords for a single word.
  Intended for use with collapse_subword_values"""
  return ''.join(map(lambda s: s.lstrip('#'), subword_vals))


def agg_first(subword_vals: List[T]) -> T:
  """Returns the first item of the given list.
  Intended for use with collapse_subword_values"""
  return subword_vals[0]


def agg_mode(subword_vals: List[T]) -> T:
  """Returns the most common value in the given list.
  Intended for use with collapse_subword_values"""
  c = Counter(subword_vals)
  return max(c, key=c.get)


def agg_mode_type_first_prefix(subword_vals: List[str]) -> str:
  """Returns the most common entity type with the first (non-O) prefix ('B' or
  'I'). Intended for use with collapse_subword_values on string labels"""
  assert subword_vals  # shouldn't be empty
  c = Counter([v[1:] for v in subword_vals])
  label_type = max(c, key=c.get)
  if label_type == '':
    return 'O'
  for v in subword_vals:
    prefix = v[0]
    if prefix != 'O':
      return prefix + label_type
  raise Exception('Should never get here!')


def collapse_subword_values(
    subword_values: List[T],
    alignment: Union[Sequence[int], np.ndarray],
    agg_fn: Callable[[List[T]], T] = agg_mode) -> List[T]:
  """Aggregates subword values so that the aggregated values correspond to the
  original words as specified by an alignment list.  The alignment list
  contains a subword index for each original word.

  For example, if the original token sequence was

    "john", "johanson's", "house" [len=3]

  and the subword sequence was

    "john", "johan", "##son", "'", "s", "house" [len=6]

  then the alignment would be

    0, 1, 5 [len=3]

  So if this method is called with the subword labels

    B-PER, I-PER, I-PER, O, I-PER, O [len=6]

  the alignment shown above, and an aggregating function that takes the most
  common label for each word's subwords, the result would be

    B-PER, I-PER, O [len=3]
  """
  n_words = len(alignment)
  result: List[T] = []
  for word_idx, subword_idx in enumerate(alignment):
    next_word_idx = word_idx + 1
    next_word_subword_idx = alignment[next_word_idx] \
        if next_word_idx < n_words else None
    if next_word_subword_idx is not None and \
        next_word_subword_idx <= subword_idx:
      raise ValueError('Invalid subword/word alignment (must be '
                       f'increasing indices): {alignment}')
    result.append(agg_fn(subword_values[subword_idx:next_word_subword_idx]))
  return result
