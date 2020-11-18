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

""" Abstract classes for building NER models. """
from typing import Dict

import tensorflow.compat.v1 as tf

from ner.hparam import HParams
from ner.registry import Registries


class Model:
  """ Abstract Model """

  def __init__(self, hparams):
    """ If `hparams` is a string it is looked up in the registry """
    if isinstance(hparams, str):
      self._hparams = Registries.hparams[hparams]
    elif isinstance(hparams, HParams):
      self._hparams = hparams
    else:
      raise ValueError(f'Unknown hparams parameter type: {type(hparams)}')

  def body(self, *, inputs: Dict[str, tf.Tensor], mode: str) -> tf.Tensor:
    """ Most of the computation happens here """
    raise NotImplementedError

  def loss(self, *, predictions, features, targets, is_training):
    """ Compute loss given predictions and targets """
    raise NotImplementedError

  def predict(self, *, inputs, params):
    """ Compute predictions given inputs """
    raise NotImplementedError

  def get_model_fn(self):
    """ Return a model function to be used by an `Estimator` """
    raise NotImplementedError

  @property
  def hp(self):
    return self._hparams


class Embedding:
  """ Embedding layer, that embeds tokens """

  def __init__(self, hparams):
    self._params = hparams

  def embed(self, *, inputs, is_training):
    raise NotImplementedError

  @property
  def hp(self):
    return self._params
