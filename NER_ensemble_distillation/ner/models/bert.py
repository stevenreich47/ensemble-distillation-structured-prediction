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

# pylint: disable=missing-docstring

""" NER models that directly hook up to BERT """

import os

from ner.bert_embedding import BERT
from ner.hparam import HParams
from ner.models.rnn_crf import bi_crf_adafactor_clipped
from ner.models.rnn_crf import bi_crf_adafactor_clipped_nodecay
from ner.models.rnn_crf import bi_crf_adam_clipped
from ner.models.rnn_crf import BiCRFModel
from ner.registry import Registries


def add_english_bert_base_to_hparams(hparams, frozen=True,
                                     regularize=False, use_l2=False):
  bert_dir = os.getenv("BERT_DIR")
  hparams.add_hparam("bert_vocab_size", 28996)
  hparams.add_hparam("bert_hidden_size", 768)
  hparams.add_hparam("bert_num_hidden_layers", 12)
  hparams.add_hparam("bert_num_attention_heads", 12)
  hparams.add_hparam("bert_intermediate_size", 3072)
  hparams.add_hparam("bert_type_vocab_size", 2)
  hparams.add_hparam("bert_frozen", frozen)
  if frozen:
    hparams.add_hparam("vars_to_opt", r"^(?!bert).*$")
  if use_l2:
    hparams.add_hparam("l2_vars", r"*bert.*$")
    hparams.add_hparam("lam", 0.001)
    hparams.add_hparam("init_ckpt",
                       f"{bert_dir}"
                       f"/bert_model.ckpt")
  hparams.add_hparam("bert_regularize", regularize)
  hparams.add_hparam("align", False)
  return hparams


# rubert and multilingual bert share these params
# may use this function for both models
def add_rubert_to_hparams(hparams: HParams, frozen=True,
                          use_l2=False,
                          regularize=False) -> HParams:
  mbert_dir = os.getenv("MBERT_DIR")
  hparams.add_hparam("bert_vocab_size", 119547)
  hparams.add_hparam("bert_hidden_size", 768)
  hparams.add_hparam("bert_num_hidden_layers", 12)
  hparams.add_hparam("bert_num_attention_heads", 12)
  hparams.add_hparam("bert_intermediate_size", 3072)
  hparams.add_hparam("bert_type_vocab_size", 2)
  hparams.add_hparam("bert_frozen", frozen)
  if frozen:
    hparams.add_hparam("vars_to_opt", r"^(?!bert).*$")
  if use_l2:
    hparams.add_hparam("l2_vars", r"*bert.*$")
    hparams.add_hparam("lam", 0.001)
    hparams.add_hparam("init_ckpt",
                       f"{mbert_dir}/" +
                       "bert_model.ckpt")
  hparams.add_hparam("bert_regularize", regularize)
  hparams.add_hparam("align", False)
  return hparams


@Registries.hparams.register
def bert_frozen_chinese_base_bi_crf_adafactor_clipped():
  hps = add_chinese_bert_base_to_hparams(bi_crf_adafactor_clipped(),
                                         frozen=True,
                                         regularize=False)
  hps.set_hparam('optimizer', 'adafactor')
  hps.add_hparam('use_bert_layers', '-1,-2,-3,-4')
  return hps


@Registries.hparams.register
def bert_frozen_english_base_bi_crf_adafactor_clipped():
  hps = add_english_bert_base_to_hparams(bi_crf_adafactor_clipped(),
                                         frozen=True,
                                         regularize=False)
  hps.set_hparam('optimizer', 'adafactor')
  hps.add_hparam('use_bert_layers', '-1,-2,-3,-4')
  return hps


@Registries.hparams.register
def bert_english_base_bi_crf_adafactor_clipped():
  hps = add_english_bert_base_to_hparams(bi_crf_adafactor_clipped(),
                                         frozen=False,
                                         regularize=False,
                                         use_l2=True)
  hps.add_hparam('use_bert_layers', '-1,-2,-3,-4')
  hps.set_hparam('align', True)
  hps.set_hparam('learning_rate', 0.001)
  hps.add_hparam("vars_to_opt", r"bert*|^(?!bert).*$")  # trainable variables
  hps.set_hparam('l2_vars', r"bert*")
  hps.set_hparam('lam', 0.01)
  hps.set_hparam('use_lrate_none', False)
  return hps


@Registries.hparams.register
def bert_frozen_rubert_bi_crf_adafactor_clipped():
  hps = add_rubert_to_hparams(bi_crf_adafactor_clipped(),
                              frozen=True,
                              regularize=False)
  hps.add_hparam('use_bert_layers', '-1,-2,-3,-4')
  return hps


@Registries.hparams.register
def bert_frozen_rubert_bi_crf_adafactor_clipped_aligned():
  hps = add_rubert_to_hparams(bi_crf_adafactor_clipped(),
                              frozen=True,
                              regularize=False)
  hps.add_hparam('use_bert_layers', '-1,-2,-3,-4')
  hps.set_hparam('align', True)
  hps.set_hparam('use_lrate_none', True)
  return hps


@Registries.hparams.register
def bert_frozen_rubert_bi_crf_adafactor_clipped_aligned_nodecay():
  hps = add_rubert_to_hparams(bi_crf_adafactor_clipped_nodecay(),
                              frozen=True,
                              regularize=False)
  hps.add_hparam('use_bert_layers', '-1,-2,-3,-4')
  hps.set_hparam('align', True)
  hps.set_hparam('use_lrate_none', True)
  return hps


@Registries.hparams.register
def bert_frozen_rubert_bi_crf_adafactor_windowed() -> HParams:
  hps = add_rubert_to_hparams(bi_crf_adafactor_clipped_nodecay(),
                              frozen=True,
                              regularize=True)
  hps.add_hparam('use_bert_layers', '-1,-2,-3,-4')
  hps.set_hparam('align', False)
  hps.set_hparam('use_lrate_none', True)
  return hps


@Registries.hparams.register
def bert_frozen_english_bi_crf_adafactor_clipped_aligned_nodecay():
  hps = add_english_bert_base_to_hparams(bi_crf_adafactor_clipped_nodecay(),
                                         frozen=True,
                                         regularize=False)
  hps.add_hparam('use_bert_layers', '-1,-2,-3,-4')
  hps.set_hparam('align', True)
  hps.set_hparam('use_lrate_none', True)
  return hps


@Registries.hparams.register
def bert_english_bi_crf_adafactor_clipped_aligned_nodecay():
  hps = add_english_bert_base_to_hparams(bi_crf_adafactor_clipped_nodecay(),
                                         frozen=False,
                                         regularize=False,
                                         use_l2=True)
  hps.add_hparam('use_bert_layers', '-1,-2,-3,-4')
  hps.set_hparam('align', True)
  hps.set_hparam('learning_rate', 0.001)
  hps.add_hparam("vars_to_opt", r"bert*|^(?!bert).*$")  # trainable variables
  hps.set_hparam('l2_vars', r"bert*")
  hps.set_hparam('lam', 0.01)
  hps.set_hparam('use_lrate_none', False)
  return hps


@Registries.hparams.register
def bert_multi_bi_crf_adafactor_clipped_aligned_nodecay():
  hps = add_rubert_to_hparams(bi_crf_adafactor_clipped_nodecay(),
                              frozen=False,
                              regularize=False,
                              use_l2=True)
  hps.add_hparam('use_bert_layers', '-1,-2,-3,-4')
  hps.set_hparam('align', True)
  hps.set_hparam('learning_rate', 0.0001)
  hps.add_hparam("vars_to_opt", r"bert*|^(?!bert).*$")  # trainable variables
  hps.set_hparam('l2_vars', r"bert*")
  hps.set_hparam('lam', 0.01)
  hps.set_hparam('use_lrate_none', False)
  return hps


@Registries.hparams.register
def bert_frozen_rubert_bi_crf_adam_clipped_aligned():
  hps = add_rubert_to_hparams(bi_crf_adam_clipped(),
                              frozen=True,
                              regularize=False)
  hps.set_hparam('optimizer', 'adam')
  hps.add_hparam('use_bert_layers', '-1,-2,-3,-4')
  hps.set_hparam('align', True)
  return hps


@Registries.hparams.register
def bert_rubert_bi_crf_adafactor_clipped_aligned():
  hps = add_rubert_to_hparams(bi_crf_adafactor_clipped(),
                              frozen=False,
                              regularize=False)
  hps.add_hparam('use_bert_layers', '-1,-2,-3,-4')
  hps.set_hparam('align', True)
  return hps


@Registries.hparams.register
def bert_rubert_bi_crf_adafactor_clipped_aligned_l2():
  hps = add_rubert_to_hparams(bi_crf_adafactor_clipped(),
                              frozen=False,
                              regularize=False,
                              use_l2=True)
  hps.add_hparam('use_bert_layers', '-1,-2,-3,-4')
  hps.set_hparam('align', True)
  hps.set_hparam('learning_rate', 0.001)
  hps.add_hparam("vars_to_opt", r"bert*|^(?!bert).*$")  # trainable variables
  hps.set_hparam('l2_vars', r"bert*")
  hps.set_hparam('lam', 0.01)
  hps.set_hparam('use_lrate_none', False)
  return hps


@Registries.hparams.register
def bert_frozen_chinese_base_bi_crf_adafactor_clipped_aligned():
  hps = add_chinese_bert_base_to_hparams(bi_crf_adafactor_clipped(),
                                         frozen=True,
                                         regularize=False)
  hps.add_hparam('use_bert_layers', '-1,-2,-3,-4')
  hps.set_hparam('align', True)
  return hps


@Registries.models.register("bert_lstm_crf")
class BERTBiCRFModel(BERT, BiCRFModel):
  """ LSTM-CRF that uses BERT for feature extraction """

  def __init__(self, hparams):
    BiCRFModel.__init__(self, hparams)
    BERT.__init__(self, hparams)
