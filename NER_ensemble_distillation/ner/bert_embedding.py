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

# pylint: disable=too-few-public-methods,no-self-use,too-many-locals
# pylint: disable=import-error


"""BERT as token embedding to be combined with the `Model` interface"""
from typing import Dict, List

import tensorflow.compat.v1 as tf

from ner.bert.modeling import BertConfig, BertModel
from ner.features import Features
from ner.model import Embedding


class BERT(Embedding):
  """ BERT Embedding """

  def __init__(self, hparams):
    Embedding.__init__(self, hparams)
    self.config = BertConfig(
        vocab_size=hparams.bert_vocab_size,
        hidden_size=hparams.bert_hidden_size,
        num_hidden_layers=hparams.bert_num_hidden_layers,
        num_attention_heads=hparams.bert_num_attention_heads,
        intermediate_size=hparams.bert_intermediate_size,
        type_vocab_size=hparams.bert_type_vocab_size)

  @staticmethod
  def align(inputs: Dict[str, tf.Tensor],
            bert_outputs: tf.Tensor) -> tf.Tensor:
    batch_range = tf.cast(tf.range(tf.shape(bert_outputs)[0]), tf.int64)
    raw_indices = inputs[Features.INPUT_WORD_ALIGNMENT.value]
    tiled_batch = tf.tile(tf.expand_dims(batch_range, axis=0),
                          [tf.shape(raw_indices)[1], 1])
    indices = tf.transpose(tf.stack([tiled_batch, tf.transpose(raw_indices)]))
    return tf.gather_nd(bert_outputs, indices)

  def add_gazetteer_feature(self, inputs, bert_outputs):
    gazetteer_matches = inputs[Features.INPUT_GAZETTEER_MATCHES.value]
    one_hots = tf.split(tf.one_hot(gazetteer_matches, depth=6),
                        num_or_size_splits=int(self.hp.num_entities), axis=1)
    concated_one_hots = tf.concat(one_hots, axis=-1)
    return tf.concat([bert_outputs, concated_one_hots], axis=-1)

  def embed(self, *,
            inputs: Dict[str, tf.Tensor],
            is_training: bool) -> tf.Tensor:
    """ Contract implementation to return embedding tensors"""
    is_sliding_window = self.hp.get('sliding_window_length', 0) > 0
    if self.hp.align:
      bert_lens: tf.Tensor = inputs[Features.BERT_INPUT_SEQUENCE_LENGTH.value]
    else:
      bert_lens = inputs[Features.INPUT_SEQUENCE_LENGTH.value]
      if not is_sliding_window:
        # We add two for the beginning and end-of-sequence symbols
        # that BERT assumes.
        bert_lens = tf.add(bert_lens, 2)

    input_ids: tf.Tensor = inputs[Features.INPUT_SYMBOLS.value]
    input_mask: tf.Tensor = tf.sequence_mask(bert_lens)

    model = BertModel(
        config=self.config,
        is_training=is_training and self.hp.bert_regularize,
        input_ids=input_ids,
        input_mask=input_mask
    )

    if "use_bert_layers" not in self.hp:
      # If no layers are specified, just take the very last layer by default.

      sequence_output = model.get_sequence_output()
      if is_sliding_window:
        return sequence_output

      if self.hp.align:
        # align to given tokenization (e.g. CoNLL)
        return self.align(inputs, sequence_output)

      # just use the last layer from BERT directly, excluding [CLS] and [SEP]
      return sequence_output[:, 1:-1]

    layers: List[tf.Tensor] = [
        model.all_encoder_layers[int(x)]
        for x in self.hp.use_bert_layers.split(',')]

    # ALTERNATE #2 bounding box features integration--don't remove until
    # an evaluation on real bounding box data has determined which integration
    # is better (the currently enabled one is in rnn_crf.py).

    # # concatenate bounding box features if present
    # if Features.BOUNDING_BOXES.value in inputs:
    #   bounding_boxes = inputs[Features.BOUNDING_BOXES.value]
    #   layers.append(bounding_boxes)

    # END ALTERNATE #2

    concat = tf.concat(layers, axis=-1)

    if is_sliding_window:
      return concat

    if not self.hp.align:
      # exclude [CLS] and [SEP]
      return concat[:, 1:-1]

    if "features" in self.hp and "gazetteers" in self.hp.features:
      return self.add_gazetteer_feature(inputs, self.align(inputs, concat))

    return self.align(inputs, concat)
