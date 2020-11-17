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

""" Utilities for handling model hyper-parameters """

from __future__ import absolute_import, division, print_function

from argparse import Namespace

import tensorflow.compat.v1 as tf
from absl import logging

from griffin.hparam import HParams
from griffin.registry import Registries


def hparams_with_defaults(hparams, **defaults):
  """ Return HParams object with defaults """
  default_hparams = HParams(**defaults)
  return update_hparams(default_hparams, hparams)


def update_hparams(hparams, new_hparams):
  """ Update existing with new hyperparameters """
  if new_hparams is None:
    return hparams

  if isinstance(new_hparams, str) and new_hparams.endswith('.json'):
    logging.info("Overriding default hparams from JSON")
    with open(new_hparams) as fh:
      hparams.parse_json(fh.read())
  elif isinstance(new_hparams, str):
    logging.info("Overriding default hparams from str:")
    hparams.parse(new_hparams)
  elif isinstance(new_hparams, dict):
    logging.info("Overriding default hparams from dict:")
    for k, val in new_hparams.items():
      if k in hparams:
        tf.logging.info("  {} -> {}".format(k, val))
        hparams.set_hparam(k, val)
  elif isinstance(new_hparams, Namespace):
    tf.logging.info("Overriding default hparams from Namespace:")
    for k, val in vars(new_hparams).items():
      if k in hparams and val is not None:
        tf.logging.info("  {} -> {}".format(k, val))
        hparams.set_hparam(k, val)
  else:
    raise ValueError(new_hparams)

  return hparams


def hparams_maybe_override(hparams: str,
                           hparams_str=None,
                           hparams_json=None) -> HParams:
  hp: HParams = Registries.hparams[hparams]
  update_hparams(hp, hparams_str)
  update_hparams(hp, hparams_json)
  return hp


def pretty_print_hparams(hparams):
  logging.info("HParams:")
  for k, v in hparams.values().items():
    if not v:
      continue
    if isinstance(v, HParams):
      v = "<HParams Instance>"
    logging.info(f"{k:30} {v:>25}")
