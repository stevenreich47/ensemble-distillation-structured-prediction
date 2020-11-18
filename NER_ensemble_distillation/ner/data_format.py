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

""" Utilities for dealing with TFRecords """
from typing import Dict, List, Union, Set, Optional, Tuple, Callable, Mapping
from typing import Iterable

import tensorflow.compat.v1 as tf

from ner.features import Features
from ner.registry import Registries

FEATURE_SHAPE: Dict[Features, List[Optional[int]]] = {
    Features.BERT_FEATURE_SEQUENCE: [None, 3072],  # FIXME shouldn't hardcode
    Features.FASTTEXT_FEATURE_SEQUENCE: [None, 300],
    Features.TARGET_SEQUENCE: [None],
    Features.INPUT_SEQUENCE_LENGTH: [],
    Features.INPUT_SYMBOLS: [None],
    Features.INPUT_GAZETTEER_MATCHES: [None],
    Features.INPUT_WORD_ALIGNMENT: [None],
    Features.BERT_INPUT_SEQUENCE_LENGTH: [],
    Features.INPUT_WORD_ALIGNMENT_LENGTH: [],
    Features.SENTENCE_ID: [],
    Features.BOUNDING_BOXES: [None, 4],
    Features.TEACHER_DISTS: [None, 81]
}

FEATURE_FORMAT: Dict[Features, object] = {
    Features.BERT_FEATURE_SEQUENCE: tf.FixedLenSequenceFeature(
        [3072], dtype=tf.float32),  # FIXME shouldn't hardcode
    Features.FASTTEXT_FEATURE_SEQUENCE: tf.FixedLenSequenceFeature(
        [300], dtype=tf.float32),
    Features.TARGET_SEQUENCE: tf.FixedLenSequenceFeature([], dtype=tf.int64),
    Features.INPUT_SEQUENCE_LENGTH: tf.FixedLenFeature([], dtype=tf.int64),
    Features.INPUT_SYMBOLS: tf.FixedLenSequenceFeature([], dtype=tf.int64),
    Features.INPUT_GAZETTEER_MATCHES: tf.FixedLenSequenceFeature(
        [], dtype=tf.int64),
    Features.INPUT_WORD_ALIGNMENT: tf.FixedLenSequenceFeature(
        [], dtype=tf.int64),
    Features.BERT_INPUT_SEQUENCE_LENGTH: tf.FixedLenFeature([], dtype=tf.int64),
    Features.INPUT_WORD_ALIGNMENT_LENGTH: tf.FixedLenFeature(
        [], dtype=tf.int64),
    Features.SENTENCE_ID: tf.FixedLenFeature([], dtype=tf.int64),
    Features.BOUNDING_BOXES: tf.FixedLenSequenceFeature([4], dtype=tf.float32),
    Features.TEACHER_DISTS: tf.FixedLenSequenceFeature([81], dtype=tf.float32)
}

FEATURE_TYPE: Dict[Features, tf.DType] = {
    Features.BERT_FEATURE_SEQUENCE: tf.float32,
    Features.FASTTEXT_FEATURE_SEQUENCE: tf.float32,
    Features.TARGET_SEQUENCE: tf.int64,
    Features.INPUT_SEQUENCE_LENGTH: tf.int64,
    Features.INPUT_SYMBOLS: tf.int64,
    Features.INPUT_GAZETTEER_MATCHES: tf.int64,
    Features.INPUT_WORD_ALIGNMENT: tf.int64,
    Features.BERT_INPUT_SEQUENCE_LENGTH: tf.int64,
    Features.INPUT_WORD_ALIGNMENT_LENGTH: tf.int64,
    Features.SENTENCE_ID: tf.int64,
    Features.BOUNDING_BOXES: tf.float32,
    Features.TEACHER_DISTS: tf.float32
}


def get_feature_shape(feature: Features) -> List[Optional[int]]:
  if feature in FEATURE_SHAPE:
    return FEATURE_SHAPE[feature]
  raise ValueError(f"{feature} shape information not defined. "
                   "Add feature shape to `FEATURE_SHAPE` dictionary.")


def get_feature_format(feature: Features) -> object:
  if feature in FEATURE_FORMAT:
    return FEATURE_FORMAT[feature]
  raise ValueError(f"{feature} format information not defined. "
                   "Add feature format to `FEATURE_FORMAT` dictionary.")


def get_feature_type(feature: Features) -> tf.DType:
  if feature in FEATURE_TYPE:
    return FEATURE_TYPE[feature]
  raise ValueError(f"{feature} type information not defined. "
                   "Add feature type to `FEATURE_TYPE` dictionary.")


def _parse(features: Union[str, Features, List], sep=',') -> Set[Features]:
  """ Parse features """
  if isinstance(features, str):
    feature_set = {Features[f] for f in features.split(sep)}
  elif isinstance(features, Features):
    feature_set = {features}
  elif isinstance(features, (set, list)):
    feature_set = set()
    for feature in features:
      if isinstance(feature, Features):
        feature_set.add(feature)
      if isinstance(feature, str):
        feature_set.add(Features[feature])
  else:
    raise ValueError(features)

  if not feature_set:
    raise ValueError(features)

  return feature_set


def _feature_type(features):
  """ Return feature types as a dict with string keys """
  return {f.value: get_feature_type(f) for f in features}


def _feature_format(features: Set[Features]) -> Dict[str, object]:
  """ Return feature formats as a dict with string keys """
  return {f.value: get_feature_format(f) for f in features}


def _feature_shape(features: Set[Features]) -> Dict[str, List[Optional[int]]]:
  """ Return feature shapes as a dict with string keys """
  return {f.value: get_feature_shape(f) for f in features}


class DataFormat:
  """ A `DataFormat` describes the `Features` of problem
  and should capture sufficient information to serialize
  and deserialize data from protobufs """

  def __init__(self, *, name: str,
               context_features: List[Features],
               sequence_features: List[Features],
               sequence_label_feature: Union[str, Features] = None,
               cased=True):
    """
    Arguments:
      name: `str`
      context_features: `str`, `set`, or `list` of features
      sequence_features: `str`, `set`, or `list` of features
      sequence_label_feature: `Feature` to predict
      cased: preserve case information (used in BERT)

    """
    self._name = name
    self._context_features: Set[Features] = _parse(context_features)
    self._sequence_features: Set[Features] = _parse(sequence_features)
    self._cased = cased
    if sequence_label_feature:
      if isinstance(sequence_label_feature, Features):
        self._label_feature = sequence_label_feature
      else:
        self._label_feature = Features[sequence_label_feature]
    else:
      raise ValueError(sequence_label_feature)

  @property
  def name(self):
    return self._name

  @property
  def context_features(self) -> Set[Features]:
    return self._context_features

  @property
  def sequence_features(self) -> Set[Features]:
    return self._sequence_features

  @property
  def label_feature(self) -> Features:
    return self._label_feature

  @property
  def features(self):
    return self.context_features | self.sequence_features

  @property
  def cased(self):
    return self._cased

  @property
  def length_feature(self) -> str:
    if Features.BERT_INPUT_SEQUENCE_LENGTH in self.features:
      return Features.BERT_INPUT_SEQUENCE_LENGTH.value
    if Features.INPUT_WORD_ALIGNMENT_LENGTH in self.features:
      return Features.INPUT_WORD_ALIGNMENT_LENGTH.value
    if Features.INPUT_SEQUENCE_LENGTH in self.features:
      return Features.INPUT_SEQUENCE_LENGTH.value
    raise ValueError("No length feature.")

  @property
  def sentence_id_feature(self):
    if Features.SENTENCE_ID.value in self.features:
      return Features.SENTENCE_ID.value
    raise ValueError("No sentence id feature.")

  @property
  def shape(self) -> Tuple[Dict[str, List[Optional[int]]], List[Optional[int]]]:
    """ 2-tuple of (dict of feature shapes, label shape) """
    label_shape: List[Optional[int]] = get_feature_shape(self.label_feature)
    feature_shape_dict = {f.value: get_feature_shape(f) for f in self.features}
    return feature_shape_dict, label_shape

  @property
  def parse_single_example_fn(self) -> \
      Callable[[tf.train.SequenceExample],
               Tuple[Dict[str, tf.Tensor], tf.Tensor]]:
    """Return a Python function to parse serialized SequenceExample
    instances
    """
    return self.parse_example_fn(batch=False)

  def parse_example_fn(self, batch=False) -> \
      Callable[[tf.train.SequenceExample],
               Tuple[Dict[str, tf.Tensor], tf.Tensor]]:
    """Return a Python function to parse serialized SequenceExample
    instances
    """

    def data_map_fn(serialized_example: tf.Tensor) \
        -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
      context_format: Dict[str, object] = _feature_format(self.context_features)
      sequence_format: Dict[str, object] = \
          _feature_format(self.sequence_features)

      parse_fn = tf.io.parse_sequence_example \
        if batch else tf.parse_single_sequence_example
      context, feature_lists = parse_fn(serialized_example,
                                        context_features=context_format,
                                        sequence_features=sequence_format)[:2]

      # Flatten the context and sequence features
      feature_dict: Dict[str, tf.Tensor] = {}
      for feature, value in context.items():
        feature_dict[feature] = value
      for feature, value in feature_lists.items():
        assert feature not in feature_dict
        feature_dict[feature] = value

      # Currently, we only support sequence features as the targets
      labels: tf.Tensor = feature_dict[self.label_feature.value]
      assert labels is not None

      return feature_dict, labels

    return data_map_fn

  def check_features(self,
                     context_features: Mapping[Features, int],
                     sequence_features: Mapping[Features, Iterable]):
    """ Sanity check """

    if Features.SENTENCE_ID not in self._context_features:
      raise ValueError("We notice you're using an old DataFormat;" +
                       "please use Dawn's new one with SENTENCE_ID!")

    feature: Features
    for feature, value in context_features.items():
      ctx_feature_type: tf.DType = get_feature_type(feature)
      if ctx_feature_type == tf.int64 and not isinstance(value, int):
        raise ValueError(feature)
      if ctx_feature_type == tf.float32 and not isinstance(value, float):
        raise ValueError(feature)
      assert feature in self._context_features

    for feature, values in sequence_features.items():
      if not isinstance(values, list):
        raise ValueError(f'The value of {feature} must be a list, '
                         f'but was a {type(values).__name__}!')
      if not values:
        continue
      value = values[0]
      if isinstance(value, list):
        value = value[0]
      seq_feature_type: tf.DType = get_feature_type(feature)
      if seq_feature_type == tf.int64 and not isinstance(value, int):
        raise ValueError(f'{feature} expected an int '
                         f'but was {type(value).__name__}!')
      if seq_feature_type == tf.float32 and not isinstance(value, float):
        raise ValueError(f'{feature} expected a float '
                         f'but was {type(value).__name__}!')
      assert feature in self._sequence_features


@Registries.data_formats.register
def bert_tokens_cased():
  return DataFormat(
      name="bert_tokens_cased",
      context_features=[Features.INPUT_SEQUENCE_LENGTH,
                        Features.SENTENCE_ID],
      sequence_features=[
          Features.INPUT_SYMBOLS, Features.TARGET_SEQUENCE],
      sequence_label_feature=Features.TARGET_SEQUENCE,
      cased=True)


@Registries.data_formats.register
def bert_tokens_with_words_cased():
  return DataFormat(
      name="bert_tokens_with_words_cased",
      context_features=[
          Features.INPUT_SEQUENCE_LENGTH,
          Features.BERT_INPUT_SEQUENCE_LENGTH,
          Features.INPUT_WORD_ALIGNMENT_LENGTH,
          Features.SENTENCE_ID],
      sequence_features=[
          Features.INPUT_WORD_ALIGNMENT,
          Features.INPUT_SYMBOLS,
          Features.TARGET_SEQUENCE],
      sequence_label_feature=Features.TARGET_SEQUENCE,
      cased=True)


@Registries.data_formats.register
def bert_tokens_cased_with_bounding_boxes():
  return DataFormat(
      name='bert_tokens_cased_with_bounding_boxes',
      context_features=[
          Features.INPUT_SEQUENCE_LENGTH,
          Features.BERT_INPUT_SEQUENCE_LENGTH,
          Features.INPUT_WORD_ALIGNMENT_LENGTH,
          Features.SENTENCE_ID],
      sequence_features=[
          Features.INPUT_WORD_ALIGNMENT,
          Features.INPUT_SYMBOLS,
          Features.TARGET_SEQUENCE,
          Features.BOUNDING_BOXES],
      sequence_label_feature=Features.TARGET_SEQUENCE,
      cased=True)


@Registries.data_formats.register
def bert_tokens_cased_with_teacher_dists():
  return DataFormat(
      name='bert_tokens_cased_with_teacher_dists',
      context_features=[
          Features.INPUT_SEQUENCE_LENGTH,
          Features.BERT_INPUT_SEQUENCE_LENGTH,
          Features.INPUT_WORD_ALIGNMENT_LENGTH,
          Features.SENTENCE_ID],
      sequence_features=[
          Features.INPUT_WORD_ALIGNMENT,
          Features.INPUT_SYMBOLS,
          Features.TARGET_SEQUENCE,
          Features.TEACHER_DISTS],
      sequence_label_feature=Features.TARGET_SEQUENCE,
      cased=True)


@Registries.data_formats.register
def bert_tokens_with_words_uncased():
  return DataFormat(
      name="bert_tokens_with_words_uncased",
      context_features=[
          Features.INPUT_SEQUENCE_LENGTH,
          Features.BERT_INPUT_SEQUENCE_LENGTH,
          Features.SENTENCE_ID],
      sequence_features=[
          Features.INPUT_WORD_ALIGNMENT,
          Features.INPUT_SYMBOLS,
          Features.TARGET_SEQUENCE],
      sequence_label_feature=Features.TARGET_SEQUENCE,
      cased=False)


@Registries.data_formats.register
def bert_tokens_with_words_cased_gazetteer():
  return DataFormat(
      name="bert_tokens_with_words_cased_gazetteer",
      context_features=[
          Features.INPUT_SEQUENCE_LENGTH,
          Features.BERT_INPUT_SEQUENCE_LENGTH,
          Features.SENTENCE_ID],
      sequence_features=[
          Features.INPUT_WORD_ALIGNMENT,
          Features.INPUT_SYMBOLS,
          Features.TARGET_SEQUENCE,
          Features.INPUT_GAZETTEER_MATCHES],
      sequence_label_feature=Features.TARGET_SEQUENCE,
      cased=True)


@Registries.data_formats.register
def bert_tokens_with_words_uncased_gazetteer():
  return DataFormat(
      name="bert_tokens_with_words_uncased_gazetteer",
      context_features=[
          Features.INPUT_SEQUENCE_LENGTH,
          Features.BERT_INPUT_SEQUENCE_LENGTH,
          Features.SENTENCE_ID],
      sequence_features=[
          Features.INPUT_WORD_ALIGNMENT,
          Features.INPUT_SYMBOLS,
          Features.TARGET_SEQUENCE,
          Features.INPUT_GAZETTEER_MATCHES],
      sequence_label_feature=Features.TARGET_SEQUENCE,
      cased=False)


@Registries.data_formats.register
def bert_entity_classify_uncased():
  return DataFormat(
      name="bert_entity_classify_uncased",
      context_features=[
          Features.BERT_INPUT_SEQUENCE_LENGTH,
          Features.SENTENCE_ID],
      sequence_features=[
          Features.INPUT_WORD_ALIGNMENT,
          Features.INPUT_SYMBOLS,
          Features.TARGET_SEQUENCE],
      sequence_label_feature=Features.TARGET_SEQUENCE,
      cased=False)


@Registries.data_formats.register
def bert_entity_classify_cased():
  return DataFormat(
      name="bert_entity_classify_cased",
      context_features=[
          Features.BERT_INPUT_SEQUENCE_LENGTH,
          Features.SENTENCE_ID],
      sequence_features=[
          Features.INPUT_WORD_ALIGNMENT,
          Features.INPUT_SYMBOLS,
          Features.TARGET_SEQUENCE],
      sequence_label_feature=Features.TARGET_SEQUENCE,
      cased=True)


@Registries.data_formats.register
def doc_bert_cased():
  return DataFormat(
      name="doc_bert_cased",
      context_features=[Features.INPUT_SEQUENCE_LENGTH,
                        Features.SENTENCE_ID],
      sequence_features=[
          Features.BERT_FEATURE_SEQUENCE, Features.TARGET_SEQUENCE],
      sequence_label_feature=Features.TARGET_SEQUENCE,
      cased=True)


@Registries.data_formats.register
def bert_cased():
  return DataFormat(
      name="bert_cased",
      context_features=[Features.INPUT_SEQUENCE_LENGTH,
                        Features.SENTENCE_ID],
      sequence_features=[
          Features.BERT_FEATURE_SEQUENCE, Features.TARGET_SEQUENCE],
      sequence_label_feature=Features.TARGET_SEQUENCE,
      cased=True)


@Registries.data_formats.register
def bert_uncased():
  return DataFormat(
      name="bert_uncased",
      context_features=[Features.INPUT_SEQUENCE_LENGTH,
                        Features.SENTENCE_ID],
      sequence_features=[
          Features.BERT_FEATURE_SEQUENCE, Features.TARGET_SEQUENCE],
      sequence_label_feature=Features.TARGET_SEQUENCE,
      cased=False)


@Registries.data_formats.register
def fasttext():
  return DataFormat(
      name="fasttext",
      context_features=[Features.INPUT_SEQUENCE_LENGTH,
                        Features.SENTENCE_ID],
      sequence_features=[Features.FASTTEXT_FEATURE_SEQUENCE,
                         Features.TARGET_SEQUENCE],
      sequence_label_feature=Features.TARGET_SEQUENCE)
