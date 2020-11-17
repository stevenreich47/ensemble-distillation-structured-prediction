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

"""Use Features to avoid hard-coding any feature names.

The Predictions `Enum` serves a similar purpose but for retrieving
predictions from the model.

"""

from enum import Enum


class Features(Enum):
  """All possible features that might be associated with an example"""

  # Mark: I *think* this class serves roughly the same purpose as TF's
  # FeatureColumn classes.  If so, should we migrate to use TF's FeatureColumn
  # classes?
  #
  # Note Nick's response: I don't think FeatureColumn is flexible enough for
  # our purposes, but I could be wrong. The main reason this class exists to
  # support going to and from TFRecords.
  #
  # TODO Investigate/consider migrating to use TF's FeatureColumn classes.

  # Sequence of BERT features.
  BERT_FEATURE_SEQUENCE = "bert_feat_seq"

  # Sequence of FastText features.
  FASTTEXT_FEATURE_SEQUENCE = "fasttext_feat_seq"

  # A flat list of input symbols (e.g. subword indices)
  INPUT_SYMBOLS = "input_symbols"

  # The length of the input sequence. This is used for either a list
  # of input symbols (e.g. subword token indices) or a list of
  # feature vectors.
  INPUT_SEQUENCE_LENGTH = "input_seq_len"

  # A sequence of targets to predict. This doesn't necessarily have
  # the same length as the inputs.
  TARGET_SEQUENCE = "targets"

  # A list of topics associated with a document
  DOC_TOPIC_LABEL = "doc_topic_label"

  # A global identifier for a particular instance. For example, a
  # sentence ID.
  WINDOW_ID = "window_id"

  # Indexed input words.
  INPUT_WORDS = "word_ids"

  # Indexed characters for each input word.
  INPUT_WORD_CHARS = "char_ids"
  # Lengths of each word, in characters
  INPUT_CHAR_LENGTHS = "char_lengths"

  # Bytes for each input word.
  INPUT_WORD_BYTES = "byte_ids"
  # Lengths of each word, in bytes
  INPUT_BYTE_LENGTHS = "byte_lengths"

  # Indexed subwords for each input word.
  INPUT_WORD_SUBWORDS = "subword_ids"
  # Lengths of each word, in subwords
  INPUT_SUBWORD_LENGTHS = "subword_lengths"

  # Gazetteer matches
  INPUT_GAZETTEER_MATCHES = "word_gazetteer_match_ids"

  # Alignment from CoNLL to BERT subwords
  INPUT_WORD_ALIGNMENT = "word_alignment"

  # Number of BERT subword tokens
  BERT_INPUT_SEQUENCE_LENGTH = "num_bert_subwords"

  # Number of WORDS (not subwords) in this window
  # (or, length of INPUT_WORD_ALIGNMENT feature)
  INPUT_WORD_ALIGNMENT_LENGTH = "num_aligned_words"

  # Sentence index to detect when sentences are artificially divided
  SENTENCE_ID = "sentence_id"

  # For NER on OCR input where each token also has bounding box coordinates;
  # each token will have 4 floats in [0,1]: (x1,y1), (x2,y2), where the first
  # xy pair is the upper left corner and the second is the lower right corner,
  # and each x/y is the fraction of the page size from the upper left corner of
  # the page. So, for example, a bounding box encompassing the entire upper
  # right quadrant of a page (unlikely in practice) would be represented as
  # [0.5, 0.0, 1.0, 0.5]
  BOUNDING_BOXES = "bounding_boxes"

  TEACHER_DISTS = "teacher_dists"

class Predictions(Enum):
  """ All possible predictions a model might produce """

  # Sometimes we might pass through the inputs
  INPUT_SYMBOLS = Features.INPUT_SYMBOLS.value

  # Predicted named-entity labels
  TAGS = "y_hat"

  # Confidences associated with predictions
  SEQUENCE_SCORE = "sequence_score"
  TOKEN_SCORES = "token_scores"
  SEGMENT_SCORES = "segment_scores"
  RAW_TOKEN_SCORES = "raw_token_scores"
  prediction_list = "pred_list"
  # Length of the prediction (to account for possible padding)
  LENGTH = "seq_length"
  TRANSITION_PARAMS = "transition_params"
  SENTENCE_ID = "sentence_id"
