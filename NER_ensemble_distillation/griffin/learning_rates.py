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

""" Registering learning rate schedulers. """

import tensorflow.compat.v1 as tf

from griffin.hparams import HParams
from griffin.registry import Registries

DEFAULT_HPARAMS = HParams()


@Registries.learning_rates.register
def noam(learning_rate, hparams=DEFAULT_HPARAMS):
  """Noam scheme learning rate decay
  learning_rate: initial learning rate. scalar.
  warmup_steps: scalar. During warmup_steps, learning rate increases
  until it reaches learning_rate.
  """
  if 'warmup_steps' not in hparams:
    hparams.add_hparam('warmup_steps', 4000)

  step = tf.cast(tf.train.get_global_step() + 1, dtype=tf.float32)

  return learning_rate * hparams.warmup_steps ** 0.5 * \
         tf.minimum(step * hparams.warmup_steps ** -1.5, step ** -0.5)


@Registries.learning_rates.register
def cosine_decay_restarts(learning_rate, hparams=DEFAULT_HPARAMS):
  """TODO describe this function"""
  if 'first_decay_steps' not in hparams:
    hparams.add_hparam('first_decay_steps', 1000)
  if 't_mul' not in hparams:
    hparams.add_hparam('t_mul', 2.0)
  if 'm_mul' not in hparams:
    hparams.add_hparam('m_mul', 1.0)
  if 'alpha' not in hparams:
    hparams.add_hparam('alpha', 0.0)

  return tf.train.cosine_decay_restarts(
      learning_rate,
      global_step=tf.train.get_global_step(),
      first_decay_steps=hparams.first_decay_steps,
      t_mul=hparams.t_mul,
      m_mul=hparams.m_mul,
      alpha=hparams.alpha)


@Registries.learning_rates.register
def cosine_decay(learning_rate, hparams=DEFAULT_HPARAMS):
  if 'decay_steps' not in hparams:
    hparams.add_hparam('decay_steps', 1000)
  if 'alpha' not in hparams:
    hparams.add_hparam('alpha', 0.0)

  return tf.train.cosine_decay(
      learning_rate,
      tf.train.get_global_step(),
      hparams.first_decay_steps,
      alpha=hparams.alpha)


@Registries.learning_rates.register
def polynomial_decay(learning_rate, hparams=DEFAULT_HPARAMS):
  """TODO describe this function"""
  if 'decay_steps' not in hparams:
    hparams.add_hparam('decay_steps', 1000)
  if 'end_learning_rate' not in hparams:
    hparams.add_hparam('end_learning_rate', 0.0001)
  if 'power' not in hparams:
    hparams.add_hparam('power', 1.0)
  if 'cycle' not in hparams:
    hparams.add_hparam('cycle', False)

  return tf.train.polynomial_decay(
      learning_rate,
      global_step=tf.train.get_global_step(),
      decay_steps=hparams.first_decay_steps,
      end_learning_rate=hparams.end_learning_rate,
      power=hparams.power,
      cycle=hparams.cycle)
