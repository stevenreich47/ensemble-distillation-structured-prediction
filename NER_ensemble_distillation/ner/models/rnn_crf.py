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

""" Recurrent neural network model with CRF output layer """
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

import tensorflow.compat.v1 as tf
import tensorflow_addons.text.crf as crf
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Concatenate

from ner.bert_regularization.finetuning_init import \
  init_vars_from_checkpoint
from ner.features import Features
from ner.features import Predictions
from ner.hparams import HParams
from ner.model import Model
from ner.optim import get_training_op
from ner.registry import Registries


@Registries.hparams.register
def bi_crf_default_9():
  return HParams(
      output_dim=9,
      birnn_layers=2,
      hidden_size=512,
      dropout_keep_prob=0.8,
      optimizer='adam',
      learning_rate=0.0001,
      beta1=0.9,
      beta2=0.999,
      use_ema=False
  )


@Registries.hparams.register
def ru_lorelei_fasttext_adafactor_crf():
  return HParams(
      output_dim=9,
      birnn_layers=1,
      hidden_size=512,
      grad_clip_norm=5.0,
      dropout_keep_prob=0.5,
      optimizer='adafactor',
      learning_rate=0.01,
      beta1=0.9,
      beta2=0.999,
      use_ema=False
  )


@Registries.hparams.register
def bi_crf_adafactor():
  return HParams(
      output_dim=9,
      birnn_layers=1,
      hidden_size=256,
      dropout_keep_prob=0.5,
      optimizer='adafactor',
      learning_rate=0.05,
      use_lrate_none=True,
      decay_rate=0.999,
      beta1=0,
      t_mul=2.0,
      m_mul=1.0,
      alpha=0.0,
      first_decay_steps=1000,
      use_ema=False,
      learning_rate_schedule='cosine_decay_restarts'
  )


@Registries.hparams.register
def bi_crf_adafactor_clipped():
  return HParams(
      output_dim=9,
      birnn_layers=1,
      hidden_size=256,
      dropout_keep_prob=0.5,
      optimizer='adafactor',
      learning_rate=0.05,
      use_lrate_none=True,
      grad_clip_norm=1.0,
      first_decay_steps=1000,
      # NOTE: This may be too high if the dataset is small.
      learning_rate_schedule='cosine_decay_restarts'
  )


@Registries.hparams.register
def bi_crf_adam_clipped():
  return HParams(
      output_dim=9,
      birnn_layers=1,
      hidden_size=256,
      dropout_keep_prob=0.5,
      optimizer='adam',
      learning_rate=0.001,
      use_lrate_none=True,
      grad_clip_norm=1.0,
      first_decay_steps=1000,
      # NOTE: This may be too high if the dataset is small.
      learning_rate_schedule='cosine_decay_restarts'
  )


@Registries.hparams.register
def bi_crf_adafactor_clipped_nodecay() -> HParams:
  return HParams(
      output_dim=9,
      birnn_layers=1,
      hidden_size=256,
      dropout_keep_prob=0.5,
      optimizer='adafactor',
      learning_rate=0.05,
      use_lrate_none=True,
      grad_clip_norm=1.0,
      # first_decay_steps=1000,
      # NOTE: This may be too high if the dataset is small.
      # learning_rate_schedule='cosine_decay_restarts'
  )


def _get_transition_params(num_tags):
  with tf.variable_scope("crf", reuse=tf.AUTO_REUSE):
    transition_params = tf.get_variable(
        "transition_params",
        [num_tags, num_tags],
        trainable=True,
        initializer=tf.random_uniform_initializer)
  return transition_params


class BiCRFModel(Model):
  """ Standard CRF model, using BiRNNs to encode at the token level.
  Override embed method to use special embeddings, i.e. BERT features.
  """

  def __init__(self, hparams):
    Model.__init__(self, hparams)
    self.transition_params = None

  def embed(self, *,
            inputs: Dict[str, tf.Tensor],
            is_training: bool) -> tf.Tensor:
    raise NotImplementedError

  def body(self, *, inputs: Dict[str, tf.Tensor], mode: str) -> tf.Tensor:
    """ Return token-level logits """

    is_training = mode == tf.estimator.ModeKeys.TRAIN

    # Get token embeddings
    embeddings = self.embed(inputs=inputs, is_training=is_training)

    # Build up sentence encoder
    rnn_keep_prob = self.hp.dropout_keep_prob if is_training else 1.
    dropout = 1. - rnn_keep_prob
    assert dropout < 1
    # model = Sequential()
    bilstm = embeddings
    for _ in range(self.hp.birnn_layers):
      lstm = LSTM(self.hp.hidden_size, return_sequences=True, dropout=dropout)
      bilstm = Bidirectional(lstm)(bilstm)

    # ALTERNATE #1 bounding box features integration--don't remove until
    # an evaluation on real bounding box data has determined which integration
    # is better (the currently disabled one is in bert_embedding.py).

    # concatenate bounding box features if present
    if Features.BOUNDING_BOXES.value in inputs:
      bounding_boxes = inputs[Features.BOUNDING_BOXES.value]
      bilstm = Concatenate()([bilstm, bounding_boxes])

    # END ALTERNATE #1

    # selu requires lecun_normal initializer
    dense = Dense(self.hp.output_dim, activation='selu',
                  kernel_initializer=tf.keras.initializers.lecun_normal())
    # dense = Dense(self.hp.output_dim, activation='gelu',
    #               kernel_initializer=tf.keras.initializers.he_normal())
    # dense = Dense(self.hp.output_dim, activation='relu',
    #               kernel_initializer=tf.keras.initializers.he_normal())
    return TimeDistributed(dense)(bilstm, is_training)

  def loss(self, *,
           predictions: tf.Tensor,
           features: Dict[str, tf.Tensor],
           targets: tf.Tensor, is_training) -> tf.Tensor:
    """ For a CRF, predictions should be token-level logits and
    targets should be indexed labels.
    """
    del is_training
    seq_lens: tf.Tensor = tf.reshape(
        features[Features.INPUT_SEQUENCE_LENGTH.value], [-1])

    # if using sliding window, slice off context zones before computing loss
    if self.hp.get('sliding_window_length', 0) > 0:
      ctx_len = self.hp.sliding_window_context
      predictions = predictions[:, ctx_len:-ctx_len, :]
      targets = targets[:, ctx_len:-ctx_len]
      seq_lens = tf.subtract(seq_lens, 2*ctx_len)

    transition_params: tf.Variable = _get_transition_params(self.hp.output_dim)
    with tf.control_dependencies([
        # for debugging, useful to uncomment these
        # tf.print("predictions: ", tf.shape(predictions)),
        # tf.print("seq_lens: ", tf.shape(seq_lens), seq_lens),
        # tf.print("targets: ", tf.shape(targets)),
        tf.assert_less(targets, tf.cast(self.hp.output_dim, tf.int64)),
    ]):
      likelihood, _ = crf.crf_log_likelihood(
          predictions,
          tf.cast(targets, tf.int32),
          tf.cast(seq_lens, tf.int32),
          transition_params=transition_params)

    batch_size = tf.shape(targets)[0]
    with tf.control_dependencies([
        tf.assert_equal(tf.shape(likelihood), batch_size, [likelihood],
                        message='likelihood tensor has wrong shape')
    ]):
      loss: tf.Tensor = tf.reduce_mean(-likelihood)

    if "l2_vars" in self.hp:
      l2_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                  self.hp.l2_vars)
      values: List[tf.Tensor] = []
      star_var_values: List[tf.Tensor] = []
      for var in l2_vars:
        values.append(tf.reshape(var.value(), [-1]))
        name = var.name[:-2]
        global_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                        "star/" + name)
        star_var_values.append(tf.reshape(global_vars[0].value(), [-1]))
      values = tf.concat(values, 0)
      star_var_values = tf.concat(star_var_values, 0)

      sum_of_squares = tf.reduce_sum(tf.square(tf.subtract(values,
                                                           star_var_values)))
      loss = tf.add(loss, tf.scalar_mul((self.hp.lam / 2), sum_of_squares))

    with tf.control_dependencies([
        tf.assert_equal(tf.shape(loss), tf.constant([], dtype='int32'),
                        [loss], message='loss tensor has wrong shape')
    ]):
      return loss

  def predict(self, *, inputs, params):
    """ Get model predictions """
    del params
    logits = self.body(inputs=inputs, mode=tf.estimator.ModeKeys.PREDICT)
    seq_lens = tf.reshape(
        inputs[Features.INPUT_SEQUENCE_LENGTH.value], [-1])
    transition_params = _get_transition_params(self.hp.output_dim)

    predictions, best_score = crf.crf_decode(logits, transition_params,
                                             seq_lens)
    return {
        Predictions.TAGS.value: predictions,
        Predictions.SEQUENCE_SCORE.value: best_score,
        Predictions.TOKEN_SCORES.value: tf.zeros_like(logits),
        Predictions.RAW_TOKEN_SCORES.value: logits
    }

  def predict_from_logits(self, *, logits, features):
    """ Do CRF decoding starting from logits, rather than raw input """
    seq_lens = tf.reshape(
        features[Features.INPUT_SEQUENCE_LENGTH.value], [-1])
    transition_params = _get_transition_params(self.hp.output_dim)
    predictions, best_score = crf.crf_decode(logits, transition_params,
                                             seq_lens)
    trans_matrix = tf.convert_to_tensor(transition_params)
    trans_matrix = tf.tile(tf.expand_dims(trans_matrix, 0),
                           [tf.shape(logits)[0], 1, 1])
    return {
        Predictions.TAGS.value: predictions,
        Predictions.SEQUENCE_SCORE.value: best_score,
        Predictions.TOKEN_SCORES.value: tf.zeros_like(logits),
        Predictions.RAW_TOKEN_SCORES.value: logits,
        Predictions.TRANSITION_PARAMS.value: trans_matrix
    }

  def get_model_fn(self) -> Callable[..., tf.estimator.EstimatorSpec]:
    """ Model function contract implementation for the RNN CRF model"""

    def fn(features: Dict[str, tf.Tensor],
           labels: Optional[tf.Tensor],
           mode: str,
           params: HParams) -> tf.estimator.EstimatorSpec:

      del params
      logits = self.body(inputs=features, mode=mode)

      star_vars = None
      if "l2_vars" in self.hp:
        star_vars = []
        for var in tf.trainable_variables(self.hp.l2_vars):
          name = var.name[:-2]
          star_vars.append(tf.get_variable("star/" + name, shape=var.shape,
                                           trainable=False))
        init_vars_from_checkpoint(self.hp.init_ckpt, "star", self.hp.l2_vars)

      if mode == tf.estimator.ModeKeys.TRAIN:
        loss = self.loss(predictions=logits,
                         features=features, targets=labels,
                         is_training=True)
        train_op = get_training_op(loss, self.hp)
        return tf.estimator.EstimatorSpec(tf.estimator.ModeKeys.TRAIN,
                                          loss=loss,
                                          train_op=train_op)

      if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = self.predict_from_logits(
            logits=logits, features=features)
        predictions[Predictions.LENGTH.value] = tf.reshape(
            features[Features.INPUT_SEQUENCE_LENGTH.value], [-1]
        )
        predictions[Predictions.SENTENCE_ID.value] = \
            features[Features.SENTENCE_ID.value]
        predictions[Features.INPUT_WORD_ALIGNMENT.value] = \
            features[Features.INPUT_WORD_ALIGNMENT.value]
        predictions[Features.INPUT_WORD_ALIGNMENT_LENGTH.value] = \
            features[Features.INPUT_WORD_ALIGNMENT_LENGTH.value]
        return tf.estimator.EstimatorSpec(tf.estimator.ModeKeys.PREDICT,
                                          predictions=predictions)

      if mode == tf.estimator.ModeKeys.EVAL:
        loss = self.loss(predictions=logits, features=features,
                         targets=labels, is_training=False)

        predictions = self.predict_from_logits(logits=logits, features=features)
        predicted_labels = predictions[Predictions.TAGS.value]

        seq_lens = tf.reshape(
            features[Features.INPUT_SEQUENCE_LENGTH.value], [-1])
        weights = tf.sequence_mask(seq_lens, dtype=tf.float32)
        accuracy = tf.metrics.accuracy(labels, predicted_labels,
                                       weights=weights)
        return tf.estimator.EstimatorSpec(
            tf.estimator.ModeKeys.EVAL, loss=loss,
            eval_metric_ops={'accuracy': accuracy})

      raise ValueError(f'Unrecognized mode: "{mode}" '
                       '(Should never reach this point!)')

    return fn


@Registries.models.register
class FeatureBiCRFModel(BiCRFModel):
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
