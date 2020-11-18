""" Utility for generating sliding windows """

from itertools import chain
from typing import Iterable, Iterator, List, TypeVar, Optional

_V = TypeVar('_V')


def sliding_window(seq: Iterable[_V], window_size: int, context_size=0,
                   padding_value: Optional[_V] = None,
                   drop_remainder=False) -> Iterator[List[_V]]:
  """Creates a sliding window view of the given Iterable

  <--------------------------- window -------------------------->
  <-- left context -><--- prediction zone ---><- right context ->
  t00  t01  t02  t03  t04  t05  t06  t07  t08  t09  t10  t11  t12

  Note that "window" is ONLY used to refer to the entire window--not any
  subset of it.  For this example, window length is 13, prediction length is 5,
  and context length is 4 (for now, we require left and right context lengths
  are the same).  So, this equation should be an invariant:

  INVARIANT: window_length = prediction_length + 2*context_length

  Only if padding_value is set, will this ALWAYS yield windows of
  length window_size. Otherwise, it may yield some windows with lengths less
  than window_size.

  :param seq: iterable source to be windowed
  :param window_size: the target size of each window (only guaranteed when
                      padding_value is not None)
  :param context_size: length of regions to include for context only
  :param padding_value: value to use for padding
  :param drop_remainder: true if you want to discard the last window if it's
                         not full
  :return: an iterator of windows (lists)
  """

  if window_size < 1:
    raise ValueError("window_size must be > 0!")

  if context_size < 0:
    raise ValueError("context_size must be >= 0!")

  prediction_size = window_size - 2 * context_size
  if prediction_size < 1:
    raise ValueError("Sliding window context is too large; window_size - "
                     "2*context_size must be > 0!")

  # before proceeding, we need to ensure the given seq is not empty
  iterator: Iterator[_V] = iter(seq)
  try:
    first_item: _V = next(iterator)
  except StopIteration:
    return

  # build padding for each end of the given seq (only if padding_value
  # is given; note that padding_value could be falsey, so check against None!)
  padding: List[_V] = [padding_value] * context_size \
      if padding_value is not None else []

  window: List[_V] = []  # the window that'll be yielded
  window_is_full = False  # true after each complete (non-partial) window

  # loop through sequence after prepending and appending padding
  for item in chain(padding, [first_item], iterator, padding):
    window.append(item)

    window_is_full = len(window) == window_size
    if window_is_full:
      yield window
      # slide the last window's contents prediction_size to the left
      window = window[prediction_size:]

  # if drop_remainder or there is no remainder (i.e., the last yielded window
  # was a full/complete one), don't worry about what's left--we're done
  if drop_remainder or window_is_full:
    return

  # if padding_value is given, pad the last window
  if padding_value is not None:
    while len(window) < window_size:
      window.append(padding_value)

  # finally, yield the last window
  yield window
