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

"""
Basic NER Data processing
  - Word and character / byte level pre-processing and data generators
  - Dataset loader
"""
import io
from typing import Callable
from typing import Iterator
from typing import List
from typing import Optional
from typing import TextIO
from typing import Union

from absl import logging


def load_conll_file(conll_file: str, max_sentence_len: Optional[int] = None) \
    -> Iterator[Union[List[List[str]], str]]:
  return load_conll(lambda: open(conll_file, 'r', encoding='utf-8'),
                    max_sentence_len)


def load_conll_string(conll_string: str,
                      max_sentence_len: Optional[int] = None) \
    -> Iterator[Union[List[List[str]], str]]:
  return load_conll(lambda: io.StringIO(conll_string), max_sentence_len)


def load_conll(text_io_factory: Callable[[], TextIO],
               max_sentence_len: Optional[int] = None) \
    -> Iterator[Union[List[List[str]], str]]:
  """ Opens a conll file and splits it up into a list of documents,
  which are lists of sentences, which are lists of tokens, which are lists of
  fields on one line of conll.
  Each sentence is a list of fields.
  Each list of fields represents one line of ConLL data,
  split up into the different items in each line
  [Token, POS, Coarse POS, ??, Tag]

  Drops sentences that exceed max_sentence_len
  """
  # a list of either (1) '-DOCSTART-'-prefixed strings OR (2) sentences
  # a list of lists of fields from each token of a sentence
  curr_sentence: List[List[str]] = []

  def check_current_sentence() -> bool:
    if not curr_sentence:
      return False
    if max_sentence_len and len(curr_sentence) > max_sentence_len:
      logging.warning(f'omitting sentence of length {len(curr_sentence)} '
                      f'that exceeds max length ({max_sentence_len})')
      return False
    return True

  with text_io_factory() as text_io:
    for line in text_io:
      stripped: str = line.strip()
      if stripped.startswith('-DOCSTART-'):
        yield stripped
      elif stripped == "":
        if check_current_sentence():
          yield curr_sentence
        curr_sentence = []
      else:
        fields: List[str] = stripped.split('\t')
        curr_sentence.append(fields)

  if check_current_sentence():
    yield curr_sentence
