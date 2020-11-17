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

""" Optimization """

import tensorflow.compat.v1 as tf
import tensorflow_addons as tfa

from griffin.registry import Registries


@Registries.optimizers.register
def adam(learning_rate, hparams):
  return tf.train.AdamOptimizer(
      learning_rate,
      beta1=hparams.get('beta1', 0.9),
      beta2=hparams.get('beta2', 0.999))


@Registries.optimizers.register
def momentum(learning_rate, hparams):
  return tf.train.MomentumOptimizer(
      learning_rate,
      momentum=hparams.get('momentum', 0.9))


@Registries.optimizers.register
def grad_descent(learning_rate, hparams):
  del hparams
  return tf.train.GradientDescentOptimizer(
      learning_rate)


@Registries.optimizers.register
def adafactor(learning_rate, hparams):
  # pylint: disable=import-outside-toplevel
  if "use_lrate_none" in hparams and hparams.use_lrate_none:
    learning_rate = None
  import griffin.adafactor as af

  del hparams
  return af.AdafactorOptimizer(learning_rate=learning_rate)


@Registries.optimizers.register
def swa(learning_rate, hparams):
  del hparams
  opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
  return tfa.optimizers.SWA(opt, start_averaging=22500, average_period=1500)


def track_params_averages():
  """ Returns EMA object and average of params """
  ema = tf.train.ExponentialMovingAverage(decay=0.9999)
  params = tf.trainable_variables()
  params_averages_op = ema.apply(params)
  return ema, params_averages_op


def get_training_op(loss, hparams):
  """ Returns op based on given hparams """
  lr = tf.constant(hparams.get("learning_rate", 1.0), dtype=tf.float32)
  if "learning_rate_schedule" in hparams:
    lr = Registries.learning_rates[hparams.learning_rate_schedule](
        learning_rate=lr, hparams=hparams)
  lr = tf.identity(lr, name="learning_rate")
  optimizer = Registries.optimizers[hparams.optimizer](lr, hparams)

  var_list = None
  if "vars_to_opt" in hparams:
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                 hparams.vars_to_opt)

  grads = optimizer.compute_gradients(loss, var_list=var_list)
  if "grad_clip_norm" in hparams:
    grads = [(tf.clip_by_value(grad,
                               -hparams.grad_clip_norm,
                               hparams.grad_clip_norm), var)
             for grad, var in grads if grad is not None]
  assert grads
  train_op = optimizer.apply_gradients(grads,
                                       global_step=tf.train.get_global_step())
  return train_op


class RestoreParametersAverageValues(tf.train.SessionRunHook):
  def __init__(self, ema):
    super(RestoreParametersAverageValues, self).__init__()
    self._ema = ema
    self._restore_ops = None

  def begin(self):
    ema_variables = tf.moving_average_variables()
    self._restore_ops = [
        tf.assign(x, self._ema.average(x)) for x in ema_variables]
