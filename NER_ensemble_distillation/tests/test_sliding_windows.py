from typing import List, Iterator

import pytest

from ner.sliding_windows import sliding_window


def test_small_prediction_zone():
  windows: List[List[int]] = list(sliding_window(
      range(3), window_size=5, context_size=2, padding_value=-1))
  assert windows == [[-1, -1, 0, 1, 2],
                     [-1, 0, 1, 2, -1],
                     [0, 1, 2, -1, -1]]


def test_large_prediction_zone():
  windows: List[List[int]] = list(sliding_window(
      range(6), window_size=6, context_size=1, padding_value=-1))
  assert windows == [[-1, 0, 1, 2, 3, 4],
                     [3, 4, 5, -1, -1, -1]]


def test_large_window():
  windows: List[List[int]] = list(sliding_window(
      range(16), window_size=6, context_size=1, padding_value=-1))
  assert windows == [[-1, 0, 1, 2, 3, 4],
                     [3, 4, 5, 6, 7, 8],
                     [7, 8, 9, 10, 11, 12],
                     [11, 12, 13, 14, 15, -1]]


def test_partial_last_prediction_zone():
  windows: List[List[int]] = list(sliding_window(
      range(3), window_size=6, context_size=2, padding_value=-1))
  assert windows == [[-1, -1, 0, 1, 2, -1],
                     [0, 1, 2, -1, -1, -1]]


def test_no_padding():
  seq: Iterator[int] = range(3)
  windows: List[List[int]] = list(sliding_window(
      seq, window_size=6, context_size=2, padding_value=None))
  assert windows == [[0, 1, 2]]


def test_drop_remainder():
  windows: List[List[int]] = list(sliding_window(
      range(3), window_size=6, context_size=2, padding_value=-1,
      drop_remainder=True))
  assert windows == [[-1, -1, 0, 1, 2, -1]]


def test_with_falsey_padding():
  windows: List[List[str]] = list(sliding_window(
      'abc', window_size=6, context_size=2, padding_value=''))
  assert windows == [['', '', 'a', 'b', 'c', ''],
                     ['a', 'b', 'c', '', '', '']]


def test_context_too_big():
  with pytest.raises(ValueError):
    next(sliding_window([1, 2, 3], window_size=5, context_size=3))


def test_too_small():
  with pytest.raises(ValueError):
    next(sliding_window([1, 2, 3], window_size=0))


def test_context_too_small():
  with pytest.raises(ValueError):
    next(sliding_window([1, 2, 3], window_size=10, context_size=-1))


def test_without_context():
  windows: List[List[int]] = list(sliding_window(
      range(5), window_size=2, context_size=0))
  assert windows == [[0, 1], [2, 3], [4]]
  windows2: List[List[int]] = list(sliding_window(
      range(5), window_size=2, context_size=0, drop_remainder=True))
  assert windows2 == [[0, 1], [2, 3]]
  windows3: List[List[int]] = list(sliding_window(
      range(5), window_size=2, context_size=0, padding_value=-1))
  assert windows3 == [[0, 1], [2, 3], [4, -1]]


def test_with_empty_source():
  assert list(sliding_window(
      [], window_size=6)) == []
  assert list(sliding_window(
      [], window_size=6, context_size=2)) == []
  assert list(sliding_window(
      [], window_size=6, drop_remainder=True)) == []
  assert list(sliding_window(
      [], window_size=6, context_size=2, padding_value=-1)) == []
