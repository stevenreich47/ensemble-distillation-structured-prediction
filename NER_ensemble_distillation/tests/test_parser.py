from typing import List

from griffin.parser import CoNLLSlidingWindowFeatureParser
from griffin.parser import TaggedStr

windowed_align = CoNLLSlidingWindowFeatureParser._align_tokenization


def _split_even(word: str) -> List[str]:
  """ A dummy tokenizer for testing """
  n = len(word)
  return [word] if n % 2 else [word[:n//2], word[n//2:]]


def test_windowed_alignment():
  tagged_words = [
      TaggedStr('Janice', 'B-PER'),
      TaggedStr('kicked', 'O'),
      TaggedStr('Bob', 'B-PER'),
      TaggedStr('Woolworth', 'I-PER'),
  ]
  expected = [
      TaggedStr('Jan', 'B-PER'),
      TaggedStr('ice', None),
      TaggedStr('kic', 'O'),
      TaggedStr('ked', None),
      TaggedStr('Bob', 'B-PER'),
      TaggedStr('Woolworth', 'I-PER'),
  ]
  results = list(windowed_align(tagged_words, _split_even,
                                max_subword_per_token=100))
  print('\n'.join([str(r) for r in results]))
  assert results == expected


def test_windowed_alignment_too_many_subwords():

  def split_all(word: str) -> List[str]:
    return list(word)

  tagged_words = [
      TaggedStr('Janice', 'B-PER'),
      TaggedStr('kicked', 'O'),
      TaggedStr('Bob', 'B-PER'),
      TaggedStr('Woolworth', 'I-PER'),
  ]
  expected = [
      TaggedStr('J', 'B-PER'),
      TaggedStr('a'),
      TaggedStr('n'),
      TaggedStr('k', 'O'),
      TaggedStr('i'),
      TaggedStr('c'),
      TaggedStr('B', 'B-PER'),
      TaggedStr('o'),
      TaggedStr('b'),
      TaggedStr('W', 'I-PER'),
      TaggedStr('o'),
      TaggedStr('o'),
  ]
  results = list(windowed_align(tagged_words, split_all,
                                max_subword_per_token=3))
  print('\n'.join([str(r) for r in results]))
  assert results == expected
