from typing import List

import pytest

from ner.parser import CoNLLAlignedSentenceFeatureParser, AlignedBertTokens

divide_tokens = CoNLLAlignedSentenceFeatureParser._divide_tokens


@pytest.fixture
def aligned() -> AlignedBertTokens:
  return AlignedBertTokens(
      alignment=[1, 2, 4, 6],
      bert_token_ids=[0, 2, 3, 4, 5, 6, 7, 1]
  )


def test_divide_max_9(aligned):
  output: List[AlignedBertTokens] = divide_tokens(aligned, max_sentence_len=9)
  assert [aligned] == output


def test_divide_max_8(aligned):
  output: List[AlignedBertTokens] = divide_tokens(aligned, max_sentence_len=8)
  assert len(output) == 2

  sent = output[0]
  assert sent.alignment == [1, 2]
  assert sent.bert_token_ids == [0, 2, 3, 4, 1]

  sent = output[1]
  assert sent.alignment == [1, 3]
  assert sent.bert_token_ids == [0, 5, 6, 7, 1]


def test_divide_max_5(aligned):
  output: List[AlignedBertTokens] = divide_tokens(aligned, max_sentence_len=5)
  assert len(output) == 4

  sent = output[0]
  assert sent.alignment == [1]
  assert sent.bert_token_ids == [0, 2, 1]

  sent = output[1]
  assert sent.alignment == [1]
  assert sent.bert_token_ids == [0, 3, 4, 1]

  sent = output[2]
  assert sent.alignment == [1]
  assert sent.bert_token_ids == [0, 5, 6, 1]

  sent = output[3]
  assert sent.alignment == [1]
  assert sent.bert_token_ids == [0, 7, 1]


def test_divide_max_sent_len_too_small(aligned):
  with pytest.raises(ValueError) as e:
    divide_tokens(aligned, max_sentence_len=4)
  assert str(e.value) == "ERROR: too many subwords in token given " \
                         "max sentence length"
