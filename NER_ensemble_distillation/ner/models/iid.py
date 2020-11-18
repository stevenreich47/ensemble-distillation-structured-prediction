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

""" Simple network that takes in word embeddings and links up to a dense layer
For calibration purposes """

# pylint: disable=no-name-in-module,no-self-use,assignment-from-no-return
# pylint: disable=inconsistent-return-statements,invalid-name, import-error
# pylint: disable=duplicate-code, trailing-whitespace

from typing import Callable

import tensorflow.compat.v1 as tf

from ner.registry import Registries
from ner.hparams import HParams
from ner.features import Features
from ner.features import Predictions
from ner.model import Model
from ner.optim import get_training_op
from ner.bert_embedding import BERT
from ner.models.bert import add_english_bert_base_to_hparams
from ner.models.bert import add_rubert_to_hparams

@Registries.hparams.register
def iid_default():
  return HParams(
      output_dim=9,
      optimizer='adafactor',
      grad_clip_norm=1.0,
      learning_rate=0.0001,
      use_lrate_none=True,
      teachers=0,
      label_smoothing=0.0,
      anchor=False
  )


@Registries.hparams.register
def bert_unfrozen_iid_english():
  hps = iid_default()
  hps = add_english_bert_base_to_hparams(hps, frozen=False,
                                         regularize=False)
  hps.add_hparam('use_bert_layers', '-1,-2,-3,-4')
  hps.set_hparam('align', True)
  hps.set_hparam('use_lrate_none', False)
  return hps


@Registries.hparams.register
def bert_unfrozen_iid_multi():
  hps = iid_default()
  hps = add_rubert_to_hparams(hps, frozen=False,
                              regularize=False)
  hps.add_hparam('use_bert_layers', '-1,-2,-3,-4')
  hps.set_hparam('align', True)
  hps.set_hparam('use_lrate_none', False)
  return hps


class IIDModel(Model):
  """ Simple IID model that takes in feature vectors and predicts
  Dense layer
  """

  def __init__(self, hparams):
    Model.__init__(self, hparams)
    self.scaffold = None
    self.transition_params = None

  def embed(self, *, inputs, is_training):
    raise NotImplementedError

  def body(self, *, inputs, mode):
    """ Return token-level logits """

    is_training = mode == tf.estimator.ModeKeys.TRAIN

    # Get token embeddings
    token_embeddings = self.embed(inputs=inputs, is_training=is_training)
    # convert encoded inputs to logits
    #l2_scale = self.hp.get('l2_scale', 0.0005)
    logits = tf.layers.dense(token_embeddings, self.hp.output_dim)
    return logits

  def loss(self, *, predictions, features, targets, is_training):
    """ For a CRF, predictions should be token-level logits and
    targets should be indexed labels.
    """
    targets = tf.one_hot(targets, self.hp.output_dim)
    seq_lens = tf.reshape(
        features[Features.INPUT_SEQUENCE_LENGTH.value], [-1])
    weights = tf.sequence_mask(seq_lens, dtype=tf.float32)
    losses = 0.2 * tf.losses.softmax_cross_entropy(
        onehot_labels=targets,
        logits=predictions,
        reduction="none",
        label_smoothing=self.hp.label_smoothing)
    if self.hp.get('teachers', 0) > 0 and is_training:
      teachers = features[Features.TEACHER_DISTS.value]
      for i in range(self.hp.teachers):
        losses += tf.losses.softmax_cross_entropy(
            onehot_labels=teachers[:, :, \
                self.hp.output_dim*i:self.hp.output_dim*(i+1)],
            logits=predictions,
            reduction="none") / self.hp.teachers
        losses = losses * weights
    loss = tf.reduce_mean(tf.reduce_sum(losses, axis=1))
    return loss

  def predict(self, *, inputs, params):
    """ Get model predictions from inputs """
    del params
    logits = self.body(inputs=inputs, mode=tf.estimator.ModeKeys.PREDICT)
    log_probs = tf.nn.log_softmax(logits, -1)
    return {
        Predictions.TAGS.value: tf.argmax(logits, axis=-1),
        Predictions.SEQUENCE_SCORE.value: tf.reduce_sum(
            log_probs, axis=-1),
        Predictions.TOKEN_SCORES.value: log_probs,
        Predictions.RAW_TOKEN_SCORES.value: logits
    }

  def predict_from_logits(self, *, logits, features):
    """ Get model predictions from logits """
    del features
    log_probs = tf.nn.log_softmax(logits, -1)
    return {
        Predictions.TAGS.value: tf.argmax(logits, axis=-1),
        Predictions.SEQUENCE_SCORE.value: tf.reduce_sum(
            log_probs, axis=-1),
        Predictions.TOKEN_SCORES.value: log_probs,
        Predictions.RAW_TOKEN_SCORES.value: logits
    }

  def get_model_fn(self) -> Callable[..., tf.estimator.EstimatorSpec]:
    def fn(features, labels, mode, params):

      del params
      is_training = mode == tf.estimator.ModeKeys.TRAIN
      logits = self.body(inputs=features, mode=mode)

      if is_training:
        loss = self.loss(predictions=logits,
                         features=features, targets=labels,
                         is_training=is_training)

        # Using Registry for optimizers
        train_op = get_training_op(loss, self.hp)

        return tf.estimator.EstimatorSpec(mode, loss=loss,
                                          train_op=train_op,
                                          scaffold=self.scaffold)

      if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = self.predict_from_logits(logits=logits,
                                               features=features)
        predictions[Predictions.LENGTH.value] = tf.reshape(
            features[Features.INPUT_SEQUENCE_LENGTH.value], [-1]
        )
        predictions[Predictions.SENTENCE_ID.value] = \
            features[Features.SENTENCE_ID.value]
        return tf.estimator.EstimatorSpec(
            mode, predictions=predictions)

      if mode == tf.estimator.ModeKeys.EVAL:
        loss = self.loss(predictions=logits, features=features,
                         targets=labels, is_training=is_training)

        predictions = self.predict_from_logits(
            logits=logits, features=features)

        seq_lens = \
            tf.reshape(features[Features.INPUT_SEQUENCE_LENGTH.value], [-1])
        weights = tf.sequence_mask(seq_lens, dtype=tf.float32)

        predicted_labels = predictions[Predictions.TAGS.value]

        eval_metrics = {
            'accuracy': tf.metrics.accuracy(labels,
                                            predicted_labels,
                                            weights=weights)
        }

        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            eval_metric_ops=eval_metrics)

    return fn


@Registries.models.register("feature_iid")
class FeatureIIDModel(IIDModel):
  """ Version of BiCRF that takes in pre-computed features,
  e.g. BERT or FastText embeddings """

  def embed(self, *, inputs, is_training):
    """Nothing to embed; pass `inputs` through as-is"""
    del is_training
    if Features.BERT_FEATURE_SEQUENCE.value in inputs:
      return inputs[Features.BERT_FEATURE_SEQUENCE.value]
    if Features.FASTTEXT_FEATURE_SEQUENCE.value in inputs:
      return inputs[Features.FASTTEXT_FEATURE_SEQUENCE.value]
    raise ValueError(inputs)


@Registries.models.register("bert_iid")
class DenseBERTModel(BERT, IIDModel):

  def __init__(self, hparams):
    IIDModel.__init__(self, hparams)
    BERT.__init__(self, hparams)
