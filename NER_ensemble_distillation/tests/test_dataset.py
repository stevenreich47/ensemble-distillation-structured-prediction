from pathlib import Path

from ner.dataset import Dataset
from ner.dataset import dataset_from_file
from ner.dataset import dataset_from_string


def test_basic():
  test_conll = '\n'.join(['This	1', 'is	2', 'test	3', '',
                          'Second	4', 'test	5', '',
                          'Third	6', 'test	7'])
  dataset: Dataset = dataset_from_string(test_conll,
                                         use_bounding_boxes=True)
  assert dataset.num_documents == 1
  assert dataset.num_sentences == 3
  assert dataset.num_words == 7
  assert dataset.documents[0].num_sentences == 3
  assert dataset.documents[0].num_words == 7
  assert dataset.documents[0].doc_label_list is None
  assert dataset.documents[0].sentences[1].id == 1
  assert dataset.documents[0].sentences[1].num_words() == 2
  assert dataset.documents[0].sentences[1].tags == ['4', '5']
  assert dataset.documents[0].sentences[1].words == ['Second', 'test']
  assert dataset.documents[0].sentences[2].id == 2


def test_with_docstart():
  test_file: Path = Path(__file__).parent / 'test_load_conll_data.txt'
  dataset: Dataset = dataset_from_file(str(test_file))
  assert dataset.num_documents == 2
  assert dataset.documents[0].num_sentences == 2
  assert dataset.documents[0].num_words == 5
  assert dataset.documents[0].doc_label_list is None
  assert dataset.documents[0].sentences[1].id == 1
  assert dataset.documents[0].sentences[1].num_words() == 2
  assert dataset.documents[0].sentences[1].tags == ['4', '5']
  assert dataset.documents[0].sentences[1].words == ['Second', 'test']
  assert dataset.documents[1].num_sentences == 1
  assert dataset.documents[1].num_words == 2
  assert dataset.documents[1].doc_label_list == ['with some']
  assert dataset.documents[1].sentences[0].id == 2
  assert dataset.num_sentences == 3
  assert dataset.num_words == 7


def test_with_bounding_boxes():
  test_conll = '\n'.join(['This	1	0.10	0.2	0.3	0.4',
                          'is	2	0.11	0.2	0.3	0.4',
                          'test	3	0.12	0.2	0.3	0.4',
                          '',
                          'Second	4	0.13	0.2	0.3	0.4',
                          'test	5	0.14	0.2	0.3	0.4',
                          '',
                          '-DOCSTART-\twith some\tother junk',
                          '',
                          'Third	6	0.15	0.2	0.3	0.4',
                          'test	7	0.16	0.2	0.3	0.4'])
  dataset: Dataset = dataset_from_string(test_conll,
                                         use_bounding_boxes=True)
  assert dataset.num_documents == 2
  assert dataset.documents[0].num_sentences == 2
  assert dataset.documents[0].num_words == 5
  assert dataset.documents[0].doc_label_list is None
  assert dataset.documents[0].sentences[1].id == 1
  assert dataset.documents[0].sentences[1].num_words() == 2
  assert dataset.documents[0].sentences[1].tags == ['4', '5']
  assert dataset.documents[0].sentences[1].words == ['Second', 'test']
  assert dataset.documents[0].sentences[1].bounding_boxes == [
      [0.13, 0.2, 0.3, 0.4],
      [0.14, 0.2, 0.3, 0.4],
  ]
  assert dataset.documents[1].num_sentences == 1
  assert dataset.documents[1].num_words == 2
  assert dataset.documents[1].doc_label_list == ['with some']
  assert dataset.documents[1].sentences[0].id == 2
  assert dataset.num_sentences == 3
  assert dataset.num_words == 7
