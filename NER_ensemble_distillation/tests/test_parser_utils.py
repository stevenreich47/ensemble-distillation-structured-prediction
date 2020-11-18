import pytest

from ner.parser_utils import agg_mode_type_first_prefix
from ner.parser_utils import collapse_subword_values
from ner.parser_utils import agg_concat_bert, agg_first, agg_mode


def test_collapse_subword_values_invalid_alignment_order():
  subword_values = ["B-PER", "I-PER", "I-PER", "O", "I-PER", "O"]
  alignment = [1, 0, 5]
  with pytest.raises(ValueError):
    collapse_subword_values(subword_values, alignment, agg_mode)


def test_collapse_subword_values_doc_example():
  subword_values = ["B-PER", "I-PER", "I-PER", "O", "I-PER", "O"]
  alignment = [0, 1, 5]
  assert collapse_subword_values(
      subword_values, alignment, agg_mode) == ["B-PER", "I-PER", "O"]


def test_collapse_subword_values_doc_example_2():
  subword_values = ["B-PER", "O", "I-PER", "B-PER", "I-PER", "O"]
  alignment = [0, 1, 5]
  assert collapse_subword_values(
      subword_values, alignment, agg_mode) == ["B-PER", "I-PER", "O"]


def test_collapse_subword_values_doc_example_with_agg_first():
  subword_values = ["B-PER", "I-PER", "B-PER", "O", "B-PER", "O"]
  alignment = [0, 1, 5]
  assert collapse_subword_values(
      subword_values, alignment, agg_first) == ["B-PER", "I-PER", "O"]


def test_collapse_subwords():
  subword_values = ["john", "johan", "##son", "'", "s", "house"]
  alignment = [0, 1, 5]
  assert collapse_subword_values(
      subword_values, alignment, agg_concat_bert) == \
         ["john", "johanson's", "house"]


def test_collapse_subwords_empty():
  subword_values = ["john", "johan", "##son", "'", "s", "house"]
  alignment = []
  assert collapse_subword_values(
      subword_values, alignment, agg_concat_bert) == []


def test_collapse_subwords_padded():
  subword_values = ["john", "johan", "##son", "'", "s", "house"]
  alignment = [0, 1, 5, 0, 0, 0]
  with pytest.raises(ValueError):
    collapse_subword_values(subword_values, alignment, agg_concat_bert)


def test_agg_mode_type_first_prefix():
  assert agg_mode_type_first_prefix(['O']) == 'O'
  assert agg_mode_type_first_prefix(['O', 'O']) == 'O'
  assert agg_mode_type_first_prefix(['B-PER', 'I-PER']) == 'B-PER'
  assert agg_mode_type_first_prefix(['B-PER', 'O']) == 'B-PER'
  assert agg_mode_type_first_prefix(['O', 'B-PER']) == 'O'
  assert agg_mode_type_first_prefix([
      'B-PER', 'I-PER', 'I-ORG', 'B-ORG', 'I-ORG']) == 'B-ORG'
  assert agg_mode_type_first_prefix([
      'B-PER', 'I-PER', 'I-ORG', 'B-PER', 'I-ORG']) == 'B-PER'
  assert agg_mode_type_first_prefix([
      'I-PER', 'B-PER', 'B-ORG', 'B-ORG']) == 'I-PER'
  assert agg_mode_type_first_prefix(['B-PER', 'O', 'I-ORG', 'O', 'O']) == 'O'
  assert agg_mode_type_first_prefix(['O', 'I-ORG', 'B-PER', 'B-PER']) == 'I-PER'
