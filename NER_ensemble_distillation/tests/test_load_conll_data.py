from pathlib import Path
from typing import List

from ner.conll_io import load_conll_file


def test_with_docstart():
  test_file: str = str(Path(__file__).parent / 'test_load_conll_data.txt')
  output: List[List[List[str]]] = list(load_conll_file(test_file))
  assert len(output) == 4
  assert output[0] == [['This', '1'], ['is', '2'], ['test', '3']]
  assert output[1] == [['Second', '4'], ['test', '5']]
  assert output[2] == '-DOCSTART-\twith some\tother junk'
  assert output[3] == [['Third', '6'], ['test', '7']]
