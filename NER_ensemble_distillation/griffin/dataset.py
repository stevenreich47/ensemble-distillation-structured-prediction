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

""" Dataset objects """
import io
import itertools
import re
from typing import Callable
from typing import cast
from typing import Iterator
from typing import List
from typing import Optional
from typing import TextIO
from typing import Union

from absl import logging

from griffin.conll_io import load_conll

BASE_TAG = 'O'


class Sentence:
  """ Sentence class:
      Sentence is composed of words and the tags of those words.

      Should be able to:
          - Return words and tags
          - Return itself as a sequence of bytes and the associated tags for
            each byte, as well as spans.
  """

  def __init__(self, _id: int,
               word_list: List[str],
               tag_list: Optional[List[str]] = None,
               bounding_boxes: Optional[List[List[float]]] = None,
               teacher_dists: Optional[List[List[float]]] = None):
    self.id = _id
    self.word_list: List[str] = word_list
    self.tag_list = tag_list
    self.bounding_boxes_list = bounding_boxes
    self.teacher_dist_list = teacher_dists

  @staticmethod
  def create_sentence(sent_id: int, sentence_data: List[List[str]],
                      use_bounding_boxes=False, use_teacher_dists=False):
    """Produce sentence"""
    word_list: List[str] = [wt[0] for wt in sentence_data]
    tag_list: List[str] = [wt[1] for wt in sentence_data]
    bounding_boxes: Optional[List[List[float]]] = [
        list(map(float, wt[2:6])) for wt in sentence_data
    ] if use_bounding_boxes else None
    teacher_dists: Optional[List[List[float]]] = [
        list(itertools.chain(
            *[list(map(float, re.split(",|:", dist)[1::2])) for dist in wt[2:]]
        )) for wt in sentence_data
    ] if use_teacher_dists else None
    return Sentence(sent_id, word_list, tag_list, bounding_boxes, teacher_dists)

  def num_words(self) -> int:
    """ Number of words """
    return len(self.word_list)

  @property
  def words(self) -> List[str]:
    return self.word_list

  @property
  def tags(self) -> Optional[List[str]]:
    return self.tag_list

  @property
  def bounding_boxes(self) -> Optional[List[List[float]]]:
    return self.bounding_boxes_list

  @property
  def teacher_dists(self) -> Optional[List[List[float]]]:
    return self.teacher_dist_list

  def __repr__(self):
    return f"Sentence(Words: {self.words}\n\tTags: {self.tags})"


class Document:
  """ Document class:
      Documents are composed of a number of sentences.

      Class provides:
          - Creates Sentence Objects for each sentence in the dataset, along
            with IDs, and keeps them in one place.
          - Statistics
  """

  def __init__(self,
               doc_id: int,
               sentences: List[Sentence],
               doc_label: Optional[str] = None):

    self.doc_id: int = doc_id
    self.sentences: List[Sentence] = sentences
    self.num_words = sum(s.num_words() for s in sentences)
    self.num_sentences = len(sentences)
    self.doc_label_list = [doc_label] if doc_label else None
    self.log_document_stats()

  @staticmethod
  def create_document(doc_sent_data: List[List[List[str]]],
                      doc_id: int,
                      doc_label: str,
                      next_sent_id: int,
                      use_bounding_boxes=False,
                      use_teacher_dists=False):
    """Extract sentences from provided data."""
    sentences: List[Sentence] = []
    for idx, sent_data in enumerate(doc_sent_data):
      sent = Sentence.create_sentence(next_sent_id + idx,
                                      sent_data,
                                      use_bounding_boxes,
                                      use_teacher_dists)
      sentences.append(sent)
    return Document(doc_id, sentences, doc_label)

  def get_sentences(self) -> List[Sentence]:
    """ List all sentences in the document """
    return self.sentences

  def get_label(self) -> Optional[List[str]]:
    """ Get all labels of the document """
    return self.doc_label_list

  def log_document_stats(self) -> None:
    """ Just log some basic info """
    logging.info("Document Statistics for {}\n\
                        Total number of words: {}\n\
                        Total number of sentences: {}".format(
                            self.doc_id,
                            self.num_words,
                            self.num_sentences))


class Dataset:
  """ Dataset class:
      Datasets are composed of a number of sentences.

      *Making the assumption that sentences are the "end goal" of our NER
      systems. What we care about is writing out predictions in
      sentence-level chunks, and we don't need to break things down further
      at this level.

      Class provides:
          - Creates Sentence Objects for each sentence in the dataset, along
            with IDs, and keeps them in one place.
          - Statistics
  """

  def __init__(self, name: str, documents: List[Document]):
    self.file_name = name
    self.documents = documents
    # compute/log some stats
    self.num_documents = len(self.documents)
    self.num_sentences = sum({d.num_sentences for d in self.documents})
    self.num_words = sum({d.num_words for d in self.documents})
    self.log_dataset_stats()

  @property
  def sentence_iter(self) -> Iterator[Sentence]:
    return itertools.chain(*[d.sentences for d in self.documents])

  @property
  def sentences(self) -> List[Sentence]:
    return list(self.sentence_iter)

  def get_documents(self) -> List[Document]:
    """ Return a list of all documents in the dataset """
    return self.documents

  def log_dataset_stats(self) -> None:
    """ Just log some basic info """
    logging.info(f'Dataset Statistics for {self.file_name}\n'
                 f'Total number of words: {self.num_words}\n'
                 f'Total number of sentences: {self.num_sentences}')


def dataset_from_string(data_str: str,
                        name='from-string',
                        max_sentence_len=0,
                        use_bounding_boxes=False,
                        use_teacher_dists=False) -> Dataset:
  return _parse_dataset(lambda: io.StringIO(data_str), name, max_sentence_len,
                        use_bounding_boxes, use_teacher_dists)


def dataset_from_file(file_name: str, max_sentence_len=0,
                      use_bounding_boxes=False,
                      use_teacher_dists=False) -> Dataset:
  return _parse_dataset(lambda: open(file_name, encoding='utf-8'),
                        file_name, max_sentence_len,
                        use_bounding_boxes, use_teacher_dists)


def _parse_dataset(text_io_factory: Callable[[], TextIO],
                   name: str,
                   max_sentence_len=0,
                   use_bounding_boxes=False,
                   use_teacher_dists=False) -> Dataset:
  """Parses a CoNLL dataset"""
  documents: List[Document] = []
  next_sent_id = 0
  curr_doc_label = None

  # accumulates for each document
  curr_doc_sent_data: List[List[List[str]]] = []

  def append_doc():
    nonlocal curr_doc_sent_data, next_sent_id
    doc_id = len(documents)
    document = Document.create_document(curr_doc_sent_data, doc_id,
                                        curr_doc_label, next_sent_id,
                                        use_bounding_boxes, use_teacher_dists)
    documents.append(document)
    next_sent_id += document.num_sentences
    curr_doc_sent_data = []

  all_conll_data: Iterator[Union[List[List[str]], str]] = \
    load_conll(text_io_factory, max_sentence_len)
  for line_data in all_conll_data:
    if '-DOCSTART-' in line_data:
      # make curr_doc_sent_data into a new Document object
      if curr_doc_sent_data:  # skip creating first empty Document object
        append_doc()
      line_data = cast(str, line_data)  # a little help for the type checker
      # labels are tab separated, same line as -DOCSTART-
      docstart_fields = line_data.split('\t')
      curr_doc_label = docstart_fields[1] \
        if len(docstart_fields) > 1 else None
    else:
      line_data = cast(List[List[str]], line_data)  # help for type checker
      curr_doc_sent_data.append(line_data)

  if curr_doc_sent_data:  # create and add in last document
    append_doc()

  return Dataset(name, documents)
