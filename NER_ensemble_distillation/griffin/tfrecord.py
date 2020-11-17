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

from glob import glob
from typing import cast
from typing import Dict
from typing import Iterable
from typing import List
from typing import Mapping
from typing import Union

import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.train import Feature
from tensorflow.compat.v1.train import FeatureList
from tensorflow.compat.v1.train import FeatureLists
from tensorflow.compat.v1.train import Features
from tensorflow.compat.v1.train import FloatList
from tensorflow.compat.v1.train import Int64List
from tensorflow.compat.v1.train import SequenceExample

from griffin.data_format import get_feature_type
from griffin.features import Features as GriffinFeatures


def number_of_records(file_name: str) -> int:
  """Count number of TFRecords in `file_name`"""
  if not file_name:
    raise ValueError("Must provide valid file pattern")
  total = 0
  for path in glob(file_name):
    total += sum(1 for _ in tf.python_io.tf_record_iterator(path))
  return total


def check_tfrecord_input(pattern: str) -> None:
  if number_of_records(pattern) < 1:
    raise ValueError(f"No records in {pattern}")


def sanity_check_tfrecord(pattern, data_format, hparams, label_map):
  """ Arguments
     pattern: file pattern for tfrecords
     data_format: data format
     hparams: hyperparameters
     label_map: dictionary mapping labels to integers

  """
  # Check that the label map agrees with the hparams
  output_dim = hparams.output_dim
  if output_dim != len(label_map):
    raise ValueError(f"Model output dimensions ({output_dim}) do not "
                     f"agree with size of label map ({len(label_map)}).")

  label_set = set()
  graph = tf.Graph()
  with graph.as_default():
    data = tf.data.TFRecordDataset(
        tf.data.Dataset.list_files(pattern))
    data = data.map(data_format.parse_single_example_fn)
    iterator = data.make_one_shot_iterator()
    next_element = iterator.get_next()
    with tf.Session() as sess:
      tf.logging.info(f"Checking input files: {pattern}")
      n_examples = 0
      while True:
        try:
          ex = sess.run(next_element)
          n_examples += 1
          for label in ex[1]:
            label_set.add(label)
            if label < 0 or label >= output_dim:
              raise ValueError(
                  f"Label {label} but output dim {output_dim}")
        except tf.errors.OutOfRangeError:
          if n_examples < 1:
            raise ValueError(pattern + ' is an empty tfrecord file!')
          tf.logging.info(f"Checked {n_examples} records with no errors")
          break

      if len(label_set) != len(label_map):
        tf.logging.warning(
            f"The number of labels in the data ({len(label_set)}) "
            f"does not equal size of label map ({len(label_map)}).\n\t"
            "This can happen (and be okay) for some very small data sets, "
            "but most of the time it should not happen.\n")


def make_sequence_example_document(inputs, doc_labels):
  """Create a tf.train.SequenceExample from `inputs` and `labels` where
  inputs are embeddings of words in a document and labels are document-level"""
  input_features = [
      tf.train.Feature(float_list=tf.train.FloatList(value=input_))
      for input_ in inputs]

  return tf.train.SequenceExample(
      feature_lists=tf.train.FeatureLists(
          feature_list={
              "inputs": tf.train.FeatureList(feature=input_features)
          }),
      context=tf.train.Features(
          feature={
              "labels": tf.train.Feature(
                  int64_list=tf.train.Int64List(value=doc_labels)),
              "length": tf.train.Feature(
                  int64_list=tf.train.Int64List(value=[len(inputs)]))
          }))


def make_sequence_example(
    context_features: Mapping[GriffinFeatures, int],
    sequence_features: Mapping[GriffinFeatures, Iterable]
) -> SequenceExample:
  """ Arguments:
    context_features: Dictionary from `Features` to values.
    sequence_features: Dictionary from `Features` to values.
  """

  def _floats_to_feature_list(values: Iterable[List[float]]) -> List[Feature]:
    return [Feature(float_list=FloatList(value=value)) for value in values]

  def _ints_to_feature_list(
      values: Iterable[Union[int, List[int]]]) -> List[Feature]:
    ret = []
    for value in values:
      if isinstance(value, int):
        value = [value]
      ret.append(Feature(int64_list=Int64List(value=value)))
    return ret

  def _values_to_feature_list(
      feature: Features,
      values: Union[Iterable[Union[int, List[int]]],
                    Iterable[List[float]]]
  ) -> Feature:
    feature_type = get_feature_type(feature)
    if feature_type == tf.int64:
      return _ints_to_feature_list(
          cast(Iterable[Union[int, List[int]]], values))
    if feature_type == tf.float32:
      return _floats_to_feature_list(cast(Iterable[List[float]], values))
    raise ValueError(f'Feature {feature} has (currently) unsupported '
                     f'type: {feature_type}')

  def _int_to_feature(value: int) -> Feature:
    return Feature(int64_list=Int64List(value=[value]))

  ctx_features: Dict[str, Features] = {
      griffin_feature.value: _int_to_feature(feature_value)
      for griffin_feature, feature_value in context_features.items()
  }

  seq_feature_lists: Dict = {
      griffin_feature.value: FeatureList(feature=_values_to_feature_list(
          griffin_feature, feature_val))
      for griffin_feature, feature_val in sequence_features.items()
  }

  for feat in sequence_features.keys():
    assert feat.value in seq_feature_lists

  for feat in context_features.keys():
    assert feat.value in ctx_features

  return SequenceExample(
      context=Features(feature=ctx_features),
      feature_lists=FeatureLists(feature_list=seq_feature_lists),
  )
