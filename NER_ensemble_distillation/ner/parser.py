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

""" Parsers to serialize data files into TFRecords """
import pickle
import sys
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import TypeVar
from typing import Union

import jsonlines
from tensorflow.compat.v1.train import SequenceExample
from tqdm import tqdm

import ner.bert.tokenization as tokenization
from ner.data_format import DataFormat
from ner.dataset import Dataset
from ner.dataset import dataset_from_file
from ner.dataset import Sentence
from ner.features import Features
from ner.parser_utils import tags_to_ids
from ner.registry import Registries
from ner.sliding_windows import sliding_window
from ner.tfrecord import make_sequence_example


class Parser(ABC):
  """Abstract `Parser` object

  All subclasses must implement a `__call__` method which yields
  `SequenceExample`.
  """

  def __init__(self,
               data_format_str: str,
               label_map: str,
               vocab: str):

    if not self.supported_data_format(data_format_str):
      raise ValueError(f"Parser doesn't support {data_format_str}")
    self._data_format_str = data_format_str
    self._data_format: DataFormat = Registries.data_formats[data_format_str]()

    if not label_map:
      raise ValueError("Must provide label map file")
    with open(label_map, 'rb') as handle:
      self._label_map = pickle.load(handle)

    if not vocab:
      raise ValueError("Must provide vocab file")
    self._vocab = vocab

  @abstractmethod
  def __call__(self, kw_args) -> Iterator[SequenceExample]:
    """ Parses an input into an iterator of SequenceExamples """

  @staticmethod
  @abstractmethod
  def supported_data_format(data_format_str: str) -> bool:
    """ Checks that the given format is supported """

  @property
  def data_format(self) -> DataFormat:
    return self._data_format


class AlignedBertTokens(NamedTuple):
  """alignment will be an int -> int mapping between the
  `orig_tokens` index and the `bert_tokens` index.

  orig_tokens = ["john", "johanson", "'s", "house"]
  bert_tokens == ["[CLS]", "john", "johan", "##son", "'", "s", "house", "[SEP]"]
  bert_token_ids == [0, 12, 345, 123, 348, 111, 90, 1] # made-up IDs
  alignment == [1, 2, 4, 6]
  """
  alignment: List[int]
  bert_token_ids: List[int]


_V = TypeVar('_V')


class Tagged(NamedTuple, Generic[_V]):
  # subword's index in the subword vocabulary
  value: _V
  # entity type label ID, or None if not first subword token
  tag: Optional[_V] = None


class TaggedStr(NamedTuple):
  # subword's index in the subword vocabulary
  value: str
  # entity type label ID, or None if not first subword token
  tag: Optional[str] = None
  # optional bounding box coordinates [x1, y1, x2, y2]
  bounding_box: Optional[List[float]] = None


class TaggedId(NamedTuple):
  # subword's index in the subword vocabulary
  value: int
  # entity type label ID, or None if not first subword token
  tag: Optional[int] = None


class AlignedTaggedStr(NamedTuple):
  """alignment will be an int -> int mapping between the
  `orig_tokens` index and the `bert_tokens` index.

  orig_tokens = ["john", "johanson", "'s", "house"]
  bert_tokens == ["[CLS]", "john", "johan", "##son", "'", "s", "house", "[SEP]"]
  bert_token_ids == [0, 12, 345, 123, 348, 111, 90, 1] # made-up IDs
  alignment == [1, 2, 4, 6]
  """
  alignment: List[int]
  bert_token_ids: List[TaggedStr]


class AlignedTaggedIds(NamedTuple):
  """alignment will be an int -> int mapping between the
  `orig_tokens` index and the `bert_tokens` index.

  orig_tokens = ["john", "johanson", "'s", "house"]
  bert_tokens == ["[CLS]", "john", "johan", "##son", "'", "s", "house", "[SEP]"]
  bert_token_ids == [0, 12, 345, 123, 348, 111, 90, 1] # made-up IDs
  alignment == [1, 2, 4, 6]
  """
  alignment: List[int]
  bert_token_ids: List[TaggedId]


def _align_tokenization(
    orig_tokens: List[str],
    tokenizer: tokenization.FullTokenizer,
    max_subword_per_token: Optional[int] = None) -> AlignedBertTokens:
  """ Align BERT -> CoNLL """
  bert_tokens: List[str] = []
  # Token map will be an int -> int mapping between the
  # `orig_tokens` index and the `bert_tokens` index.
  orig_to_tok_map: List[int] = []
  bert_tokens.append("[CLS]")
  for orig_token in orig_tokens:
    orig_to_tok_map.append(len(bert_tokens))
    subwords: List[str] = tokenizer.tokenize(orig_token)
    if max_subword_per_token and len(subwords) > max_subword_per_token:
      subwords = subwords[:max_subword_per_token]
    bert_tokens.extend(subwords)
  bert_tokens.append("[SEP]")
  # orig_tokens = ["john", "johanson", "'s", "house"]
  # bert_tokens == ["[CLS]", "john", "johan", "##son", "'", "s", "house",
  #                 "[SEP]"]
  # orig_to_tok_map == [1, 2, 4, 6]
  bert_token_ids: List[int] = tokenizer.convert_tokens_to_ids(bert_tokens)
  return AlignedBertTokens(alignment=orig_to_tok_map,
                           bert_token_ids=bert_token_ids)


def _gather_bert_features(feats) -> List:
  floats = []
  for layer in feats['layers']:
    floats.extend(layer['values'])
  return floats


@Registries.parsers.register("conll_bert_tokenized")
class BERTCoNLLSentenceParser(Parser):
  """Read CoNLL sentences that have already been tokenized according to
  a given BERT vocabulary. This means that each line of the input
  CoNLL file contains one BERT subword as well as a corresponding
  label.
  """

  def __init__(self, data_format_str, label_map: str, vocab,
               max_sentence_len: Optional[int] = 510):
    Parser.__init__(self, data_format_str, label_map, vocab)
    self._do_lower_case = not self.data_format.cased
    self._max_sentence_len = max_sentence_len

  @staticmethod
  def supported_data_format(data_format_str: str) -> bool:
    return data_format_str in ["bert_tokens_cased",
                               "bert_tokens_uncased"]

  def __call__(self, conll: str):
    """ Yield SequenceExample instances from input files """

    if not conll:
      raise ValueError("Must provide CoNLL file")

    tokenizer = tokenization.FullTokenizer(
        vocab_file=self._vocab, do_lower_case=self._do_lower_case)

    dataset = dataset_from_file(conll, self._max_sentence_len)
    n_sentences = len(dataset.sentences)

    n_subwords = 0
    n_unks = 0

    for sentence_index in tqdm(range(n_sentences)):
      sentence = dataset.sentences[sentence_index]
      n_words = len(sentence.words)
      words = ['[CLS]']
      for word in sentence.words:
        n_subwords += 1
        if word in tokenizer.vocab:
          words.append(word)
        else:
          # TODO(noa): don't hardcode [UNK]
          words.append('[UNK]')
          n_unks += 1
      words.append('[SEP]')
      subword_ids = tokenizer.convert_tokens_to_ids(
          words)
      assert len(subword_ids) == n_words + 2
      assert len(sentence.tags) == n_words

      labels = tags_to_ids(sentence.tags, self._label_map)

      context_features = {
          Features.INPUT_SEQUENCE_LENGTH: len(sentence.tags)
      }
      sequence_features = {
          Features.INPUT_SYMBOLS: subword_ids,
          Features.TARGET_SEQUENCE: labels
      }
      self.data_format.check_features(context_features, sequence_features)
      yield make_sequence_example(context_features, sequence_features)

    print(f"Wrote {n_subwords} subwords ({n_unks} unknowns)")

    if n_unks:
      print("\n\tALERT! There were unknown (sub)words. This is probably "
            "no biggie.\n")


@Registries.parsers.register("conll_features")
class CoNLLSentenceFeatureParser(Parser):
  """Combine sentence-level features with NER labels in CoNLL format."""

  def __init__(self, data_format_str, label_map: str, vocab,
               max_sentence_len: Optional[int] = None):
    """ do_lower_case is BERT-specific """
    Parser.__init__(self, data_format_str, label_map, vocab)
    self._do_lower_case = not self.data_format.cased
    self._max_sentence_len = max_sentence_len

  @staticmethod
  def supported_data_format(data_format_str: str) -> bool:
    """ Data formats that this parser will provide """
    return data_format_str in ["fasttext", "bert_cased", "bert_uncased"]

  @staticmethod
  def get_fasttext_features(conll, feature_file):
    """ Read FastText features """
    sentence_features = []
    n_sentences = len(conll.sentences)
    if n_sentences < 1:
      raise ValueError(f"{n_sentences} sentences in {conll}")
    with open(feature_file, encoding="utf-8") as inf:
      for sentence_index in range(n_sentences):
        features = []
        n_words = len(conll.sentences[sentence_index].words)
        for _ in range(n_words):
          floats = [
              float(x) for x in inf.readline().rstrip().split()[1:]]
          features.append(floats)
        sentence_features.append(features)
    return sentence_features

  def get_bert_features(self, dataset, feature_file, vocab_file,
                        max_subword_per_token):
    """ Read BERT features """
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=self._do_lower_case)
    sentences = dataset.sentences
    sentence_features = []
    sent_idx = 0
    with jsonlines.open(feature_file) as reader:
      # pylint: disable=not-an-iterable
      for features in reader:
        sentence = sentences[sent_idx]
        n_words = len(sentence.words)
        subword_features = features['features']
        alignment, bert_tokens = _align_tokenization(
            sentence.words, tokenizer, max_subword_per_token)
        assert len(alignment) == n_words
        if len(bert_tokens) != len(subword_features):
          n_a = len(bert_tokens)
          n_b = len(subword_features)
          raise ValueError(f"tokenization issue {n_a} != {n_b}")
        word_features = [
            _gather_bert_features(subword_features[i])
            for i in alignment]
        sentence_features.append(word_features)
        sent_idx += 1
    assert len(sentences) == len(sentence_features)
    return sentence_features

  def __call__(self, conll: str, features=None,
               max_sentence_len=510, max_subword_per_token=None):
    """ Yield SequenceExample instances from input files """

    if not conll:
      raise ValueError("Must provide CoNLL file")

    if not features:
      raise ValueError("Must provide features file")

    if self._data_format_str == "bert" and not self._vocab:
      raise ValueError("Must provide BERT vocabulary")

    dataset = dataset_from_file(conll, self._max_sentence_len)
    n_sentences = len(dataset.sentences)

    if self._data_format_str == "fasttext":
      feature_name = Features.FASTTEXT_FEATURE_SEQUENCE
      features = self.get_fasttext_features(dataset, features)
    elif "bert" in self._data_format_str:
      feature_name = Features.BERT_FEATURE_SEQUENCE
      features = self.get_bert_features(dataset, features, self._vocab,
                                        max_subword_per_token)
    else:
      raise ValueError(self._data_format_str)

    if len(features) != len(dataset.sentences):
      raise ValueError(
          f"Number of feature sequences ({len(features)}) ",
          f"doesn't match number of sentences ({n_sentences}).")

    for sentence_index in tqdm(range(n_sentences)):
      sentence = dataset.sentences[sentence_index]
      n_words = len(sentence.words)
      assert len(sentence.tags) == n_words
      labels = tags_to_ids(sentence.tags, self._label_map)
      inputs = features[sentence_index]
      assert len(inputs) == len(labels)
      context_features = {
          Features.INPUT_SEQUENCE_LENGTH: len(inputs),
          Features.SENTENCE_ID: sentence_index
      }
      sequence_features = {
          feature_name: inputs,
          Features.TARGET_SEQUENCE: labels
      }
      self.data_format.check_features(context_features, sequence_features)
      yield make_sequence_example(context_features, sequence_features)


@Registries.parsers.register("conll_subwords_with_alignment")
class CoNLLAlignedSentenceFeatureParser(Parser):
  """Store subwords along with the given CoNLL reference tokenization."""

  def __init__(self, data_format_str: str, label_map: str, vocab,
               max_sentence_len: int = sys.maxsize,
               use_teacher_dists: bool = False):
    """ do_lower_case is BERT-specific """
    Parser.__init__(self, data_format_str, label_map, vocab)
    self._max_sentence_len = sys.maxsize \
      if max_sentence_len < 1 else int(max_sentence_len)
    self._use_teacher_dists = bool(use_teacher_dists)

  @staticmethod
  def supported_data_format(data_format_str: str) -> bool:
    """ Data formats that this parser will provide """
    return data_format_str in ["bert_tokens_with_words_cased",
                               "bert_tokens_with_words_uncased",
                               "bert_tokens_cased_with_teacher_dists"]

  def __call__(self,
               conll: str,
               max_subword_per_token=None) -> Iterator[SequenceExample]:
    """ Yield SequenceExample instances from input files """

    if not conll:
      raise ValueError("Must provide CoNLL file")

    if self._data_format_str == "bert" and not self._vocab:
      raise ValueError("Must provide BERT vocabulary")

    dataset = dataset_from_file(conll, self._max_sentence_len,
                                use_teacher_dists=self._use_teacher_dists)
    features: List[AlignedBertTokens] = \
      self._get_bert_features(dataset, self._vocab, not self.data_format.cased,
                              max_subword_per_token=64)
    assert len(features) == dataset.num_sentences
    for sentence_index, sentence in enumerate(tqdm(
        dataset.sentences, unit=' sentences',
        desc=f'Converting sents in {Path(conll).name} to examples')):
      assert len(sentence.tags) == sentence.num_words()
      labels: List[int] = tags_to_ids(sentence.tags, self._label_map)
      assert len(sentence.tags) == len(labels)
      if self._use_teacher_dists:
        dists: List[List[float]] = sentence.teacher_dists
      aligned: AlignedBertTokens = features[sentence_index]
      assert len(aligned.alignment) == len(labels)
      if len(aligned.bert_token_ids) > self._max_sentence_len:
        pseudo_sents: List[AlignedBertTokens] = self._divide_tokens(
            aligned, self._max_sentence_len)
      else:
        pseudo_sents = [aligned]  # type: ignore[no-redef]

      label_idx = 0
      align: List[int]
      bert: List[int]
      for align, bert in pseudo_sents:
        cur_labels: List[int] = labels[label_idx:label_idx + len(align)]
        if self._use_teacher_dists:
          cur_dists: List[List[float]] = dists[label_idx:label_idx + len(align)]
        label_idx += len(align)
        assert len(cur_labels) == len(align)
        context_features: Dict[Features, int] = {
            Features.BERT_INPUT_SEQUENCE_LENGTH: len(bert),
            Features.INPUT_SEQUENCE_LENGTH: len(cur_labels),
            Features.INPUT_WORD_ALIGNMENT_LENGTH: len(align),
            Features.SENTENCE_ID: sentence_index
        }
        sequence_features: Dict[Features, Iterable] = {
            Features.INPUT_WORD_ALIGNMENT: align,
            Features.INPUT_SYMBOLS: bert,
            Features.TARGET_SEQUENCE: cur_labels
        }
        if self._use_teacher_dists:
          sequence_features[Features.TEACHER_DISTS] = list(cur_dists)
        elif self._data_format_str == 'bert_tokens_cased_with_teacher_dists':
          sequence_features[Features.TEACHER_DISTS] = \
              list([[0.0 for i in range(81)] for j in range(len(align))])
        self.data_format.check_features(context_features, sequence_features)
        yield make_sequence_example(context_features, sequence_features)

  @staticmethod
  def _get_bert_features(dataset: Dataset,
                         vocab_file: str,
                         do_lower_case: bool,
                         max_subword_per_token: Optional[int]) \
      -> List[AlignedBertTokens]:
    """ Read BERT features """
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)
    sentence_features: List[AlignedBertTokens] = []
    for sentence in dataset.sentence_iter:
      n_words = len(sentence.words)
      aligned_token_ids: AlignedBertTokens = _align_tokenization(
          sentence.words, tokenizer, max_subword_per_token)
      assert len(aligned_token_ids.alignment) == n_words
      sentence_features.append(aligned_token_ids)
    assert dataset.num_sentences == len(sentence_features)
    return sentence_features

  @staticmethod
  def _divide_tokens(aligned: AlignedBertTokens, max_sentence_len: int) \
      -> List[AlignedBertTokens]:
    """ Divide sentence into pseudo sentences"""
    orig_to_tok_map: List[int] = aligned.alignment
    orig_bert_tokens: List[int] = aligned.bert_token_ids
    if len(orig_bert_tokens) < max_sentence_len:
      return [aligned]

    pseudo_sent: List[AlignedBertTokens] = []
    # max_sentence_len - 2 is used as denominator to account for the addition
    # of the required bert tokens
    target_num_sent = (len(orig_bert_tokens) //
                       (max_sentence_len - 2)) + 1
    target_num_toks = len(orig_bert_tokens) / target_num_sent
    # Exclude the required tags from target if possible
    if target_num_toks + 1 < max_sentence_len:
      target_num_toks += 1

    bert_tokens: List[int] = [orig_bert_tokens[0]]
    tok_map: List[int] = []
    for orig_tok_idx, subword_idx in enumerate(orig_to_tok_map):
      # Don't count [SEP]
      subword_cnt = len(orig_bert_tokens) - subword_idx - 1
      if orig_tok_idx + 1 < len(orig_to_tok_map):
        subword_cnt = orig_to_tok_map[orig_tok_idx + 1] - subword_idx
      # pseudo sentence must contain at least one token
      if tok_map and len(bert_tokens) + subword_cnt + 1 > target_num_toks:
        # Adding 1 to account for BERT end token
        bert_tokens.append(orig_bert_tokens[-1])
        pseudo_sent.append(AlignedBertTokens(tok_map, bert_tokens))
        bert_tokens = [orig_bert_tokens[0]]
        tok_map = []

      tok_map.append(len(bert_tokens))
      bert_tokens.extend(
          orig_bert_tokens[subword_idx: subword_idx + subword_cnt])
      if len(bert_tokens) + 1 >= max_sentence_len:
        raise ValueError("ERROR: too many subwords in token given max "
                         "sentence length")

    bert_tokens.append(orig_bert_tokens[-1])
    pseudo_sent.append(AlignedBertTokens(tok_map, bert_tokens))

    return pseudo_sent


@Registries.parsers.register("conll_sliding_window")
class CoNLLSlidingWindowFeatureParser(Parser):
  """Store subwords along with the given CoNLL reference tokenization."""

  def __init__(self, data_format_str: str,
               label_map: str,
               vocab: str,
               window_length: int = 100,
               context_length: int = 10,
               use_bounding_boxes: bool = False,
               max_subword_per_token: Optional[Union[int, str]] = None):
    """ do_lower_case is BERT-specific """
    Parser.__init__(self, data_format_str, label_map, vocab)
    self._window_length = int(window_length)
    self._context_length = int(context_length)
    self._use_bounding_boxes = bool(use_bounding_boxes)
    self._max_subword_per_token = int(max_subword_per_token)\
      if max_subword_per_token else sys.maxsize

  @staticmethod
  def supported_data_format(data_format_str: str) -> bool:
    """ Data formats that this parser will provide """
    return data_format_str in ['bert_tokens_with_words_cased',
                               'bert_tokens_with_words_uncased',
                               'bert_tokens_cased_with_bounding_boxes']

  def __call__(self, conll: str) -> Iterator[SequenceExample]:
    """ Yield SequenceExample instances from input files """

    if not conll:
      raise ValueError('Must provide CoNLL file')

    dataset = dataset_from_file(conll, self._use_bounding_boxes)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=self._vocab, do_lower_case=not self.data_format.cased)

    sentences: Iterable[Sentence] = tqdm(
        dataset.sentences, unit=' sentences', smoothing=0.05,
        desc=f'Converting sents in {Path(conll).name} to examples')
    windows: Iterator[AlignedTaggedStr] = self._get_sliding_features(
        sentences, tokenizer)
    align: List[int]
    bert: List[TaggedStr]
    for window_index, (align, bert) in enumerate(windows):
      assert len(bert) == self._window_length
      context_features: Dict[Features, int] = {
          Features.BERT_INPUT_SEQUENCE_LENGTH: len(bert),
          Features.INPUT_WORD_ALIGNMENT_LENGTH: len(align),
          Features.INPUT_SEQUENCE_LENGTH: len(bert),
          Features.SENTENCE_ID: window_index
      }
      subwords, tags, bounding_boxes = zip(*bert)
      sequence_features: Dict[Features, List] = {
          Features.INPUT_WORD_ALIGNMENT: align,
          Features.INPUT_SYMBOLS: [tokenizer.vocab[subword]
                                   for subword in subwords],
          # 'tags is not None' check in list comp is for type checker;
          # ALL tags should be not None--see check below
          Features.TARGET_SEQUENCE: [self._label_map[tag] for tag in tags
                                     if tag is not None],
      }

      # only add bounding box features if we actually have them (or just
      # the first one, for efficiency)
      if bounding_boxes[0] is not None:
        sequence_features[Features.BOUNDING_BOXES] = list(bounding_boxes)

      if len(sequence_features[Features.TARGET_SEQUENCE]) != len(bert):
        raise ValueError(
            f'TARGET_SEQUENCE feature should have length {len(bert)}, but '
            f'is {len(sequence_features[Features.TARGET_SEQUENCE])}!')
      self.data_format.check_features(context_features, sequence_features)
      yield make_sequence_example(context_features, sequence_features)
      window_index += 1

  def _get_sliding_features(self, sentences: Iterable[Sentence],
                            tokenizer: tokenization.FullTokenizer) \
      -> Iterator[AlignedTaggedStr]:
    """ Joins a list of sentences (a document) and then
    breaks them into a list of fixed-length windows.
    """
    words_n_tags: Iterator[TaggedStr] = \
      CoNLLSlidingWindowFeatureParser._sents_to_tagged_strings(sentences)
    window_features: Iterator[TaggedStr] = self._align_tokenization(
        words_n_tags, tokenizer.tokenize, self._max_subword_per_token)
    padding_box = [0., 0., 0., 0.] if self._use_bounding_boxes else None
    windows: Iterator[List[TaggedStr]] = \
        sliding_window(window_features,
                       window_size=self._window_length,
                       context_size=self._context_length,
                       padding_value=TaggedStr('-', 'O', padding_box))

    def windowed_tokens_to_ids(wins: Iterator[List[TaggedStr]]) -> \
        Iterator[AlignedTaggedStr]:
      last_label = 'O'
      for w in wins:
        alignment: List[int] = []
        subword_ids: List[TaggedStr] = []
        for i, (sw, tag, bbox) in enumerate(w):
          if tag is None:
            subword_ids.append(TaggedStr(sw, last_label, bbox))
          else:
            alignment.append(i)
            subword_ids.append(TaggedStr(sw, tag, bbox))
            last_label = 'I' + tag[1:] if tag.startswith('B-') else tag
        yield AlignedTaggedStr(alignment, subword_ids)

    return windowed_tokens_to_ids(windows)

  @staticmethod
  def _sents_to_tagged_strings(sentences: Iterable[Sentence]) \
      -> Iterator[TaggedStr]:
    """Converts a series of Sentences to a series of TaggedStr"""
    # noinspection PyPep8Naming
    T = TypeVar('T')

    def idx_val_or_none(values: Optional[List[T]], idx: int) -> Optional[T]:
      return None if values is None else values[idx]

    for sent in sentences:
      for i, word in enumerate(sent.words):
        yield TaggedStr(word,
                        idx_val_or_none(sent.tags, i),
                        idx_val_or_none(sent.bounding_boxes, i))

  @staticmethod
  def _aligned_str_to_aligned_int(aligned_str: AlignedTaggedStr,
                                  value_map: Dict[str, int],
                                  tag_map: Dict[str, int]) -> AlignedTaggedIds:
    tagged_ids: List[TaggedId] = [
        TaggedId(value_map[t.value], tag_map[t.tag] if t.tag else None)
        for t in aligned_str.bert_token_ids
    ]
    return AlignedTaggedIds(aligned_str.alignment, tagged_ids)

  @staticmethod
  def _align_tokenization(orig_tokens: Iterator[TaggedStr],
                          tokenizer: Callable[[str], List[str]],
                          max_subword_per_token: int) -> Iterator[TaggedStr]:
    """ Align BERT -> CoNLL """
    for orig_token, label, bounding_box in orig_tokens:
      subwords: List[str] = tokenizer(orig_token)

      if max_subword_per_token and len(subwords) > max_subword_per_token:
        print(f'Truncating subwords of token "{orig_token}" from '
              f'{len(subwords)} to {max_subword_per_token} subwords')
        subwords = subwords[:max_subword_per_token]

      yield(TaggedStr(subwords[0] if subwords else '[UNK]',
                      label, bounding_box))
      yield from [TaggedStr(sw, tag=None, bounding_box=bounding_box)
                  for sw in subwords[1:]]


@Registries.parsers.register("conll_document_features")
class CoNLLDocumentFeatureParser(Parser):
  """Combine document-level features with NER labels in CoNLL format."""

  def __init__(self, data_format_str, label_map: str, vocab,
               max_sentence_len: int = sys.maxsize):
    """ do_lower_case is BERT-specific """
    Parser.__init__(self, data_format_str, label_map, vocab)
    self._do_lower_case = not self.data_format.cased
    self._max_sentence_len = sys.maxsize \
      if max_sentence_len < 1 else int(max_sentence_len)

  @staticmethod
  def supported_data_format(data_format_str: str) -> bool:
    """ Data formats that this parser will provide """
    return data_format_str in ["fasttext", "doc_bert_cased"]

  @staticmethod
  def get_fasttext_features(conll, feature_file):
    """ Read FastText features """
    document_features = []
    n_documents = len(conll.documents)
    if n_documents < 1:
      raise ValueError(f"{n_documents} documents in {conll}")
    with open(feature_file, encoding="utf-8") as inf:
      for document_index in range(n_documents):
        sentence_features = []
        n_sentences = len(conll.documents[document_index].sentences)
        if n_sentences < 1:
          raise ValueError(f"{n_sentences} sentences in {conll}")
        for sentence_index in range(n_sentences):
          features = []
          n_words = len(conll.documents[document_index]
                        .sentences[sentence_index].words)
          for _ in range(n_words):
            floats = [
                float(x) for x
                in inf.readline().rstrip().split()[1:]
            ]
            features.append(floats)
          sentence_features.append(features)
        document_features.append(sentence_features)
    return document_features

  def get_bert_features(self, dataset, feature_file, vocab_file):
    """ Read BERT features """
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=self._do_lower_case)
    documents = dataset.documents
    document_features = []
    sentence_features = []
    doc_idx = 0
    sent_idx = 0

    with jsonlines.open(feature_file) as reader:
      # pylint: disable=not-an-iterable
      for features in reader:
        document = documents[doc_idx]
        sentence = document.sentences[sent_idx]
        n_words = len(sentence.words)
        subword_features = features['features']
        alignment, bert_tokens = _align_tokenization(sentence.words, tokenizer)
        assert len(alignment) == n_words
        if len(bert_tokens) != len(subword_features):
          n_a = len(bert_tokens)
          n_b = len(subword_features)
          raise ValueError(f"tokenization issue {n_a} != {n_b}")
        word_features = [
            _gather_bert_features(subword_features[i])
            for i in alignment]
        sentence_features.append(word_features)
        sent_idx += 1
        assert len(word_features) == len(sentence.words)
        if sent_idx == len(document.sentences):
          assert len(document.sentences) == len(sentence_features)
          document_features.append(sentence_features)
          sentence_features = []
          sent_idx = 0
          doc_idx += 1
    assert len(documents) == len(document_features)
    return document_features

  def __call__(self, conll: str, features=None,
               max_sentence_len=510):
    """ Yield SequenceExample instances from input files """

    if not conll:
      raise ValueError("Must provide CoNLL file")

    if not features:
      raise ValueError("Must provide features file")

    if self._data_format_str == "bert" and not self._vocab:
      raise ValueError("Must provide BERT vocabulary")

    dataset = dataset_from_file(conll, self._max_sentence_len)
    n_documents = len(dataset.documents)

    if self._data_format_str == "fasttext":
      feature_name = Features.FASTTEXT_FEATURE_SEQUENCE
      features = self.get_fasttext_features(dataset, features)
    elif "bert" in self._data_format_str:
      feature_name = Features.BERT_FEATURE_SEQUENCE
      features = self.get_bert_features(dataset, features, self._vocab)
    else:
      raise ValueError(self._data_format_str)

    if len(features) != len(dataset.documents):
      raise ValueError((
          "Number of feature sequences ({len(features)}) ",
          "doesn't match number of documents ({len(data.documents)})."))

    for document_index in tqdm(range(n_documents)):
      document = dataset.documents[document_index]

      n_sentences = len(document.sentences)

      for sentence_index in range(n_sentences):
        sentence = document.sentences[sentence_index]
        n_words = len(sentence.words)
        assert len(sentence.tags) == n_words
        labels = tags_to_ids(sentence.tags, self._label_map)
        inputs = features[document_index][sentence_index]
        assert len(inputs) == len(labels)

        context_features = {
            Features.INPUT_SEQUENCE_LENGTH: len(inputs)
        }
        sequence_features = {
            feature_name: inputs,
            Features.TARGET_SEQUENCE: labels
        }

        self.data_format.check_features(context_features, sequence_features)
        yield make_sequence_example(context_features, sequence_features)
