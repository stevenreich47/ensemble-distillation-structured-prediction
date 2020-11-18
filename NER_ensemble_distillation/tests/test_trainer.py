import numpy as np

from ner.trainer import Trainer


def test_post_process_windowed_predictions_empty():
  zone_word_gold_labels, zone_word_pred_labels = \
    Trainer._post_process_windowed_predictions(
        win_subword_gold_labels=[],
        win_subword_pred_labels=[],
        win_alignment=np.array([]),
        win_subword_len=0,
        ctx_subword_len=0,
    )
  assert zone_word_gold_labels == []
  assert zone_word_pred_labels == []


def test_post_process_windowed_predictions():
  zone_word_gold_labels, zone_word_pred_labels = \
    Trainer._post_process_windowed_predictions(
        win_subword_gold_labels=['B-PER', 'I-PER', 'I-PER',
                                 'O', 'B-ORG', 'I-ORG', 'I-ORG'],
        win_subword_pred_labels=['O', 'B-PER', 'I-PER', 'O', 'B-PER', 'I-PER', 'I-ORG'],
        win_alignment=np.array([0, 3, 4]),
        win_subword_len=7,
        ctx_subword_len=2,
    )
  assert zone_word_gold_labels == ['O', 'B-ORG']
  assert zone_word_pred_labels == ['O', 'B-PER']


def test_count_context_words_1():
  counts = Trainer._count_context_words(
      win_alignment=[0, 3, 5, 6, 9],
      win_subword_len=10,
      ctx_subword_len=3
  )
  assert counts == (1, 1)


def test_count_context_words_2():
  counts = Trainer._count_context_words(
      win_alignment=[0, 3, 5, 6, 9],
      win_subword_len=10,
      ctx_subword_len=0
  )
  assert counts == (0, 0)


def test_count_context_words_3():
  counts = Trainer._count_context_words(
      win_alignment=[0, 3, 5, 6, 9],
      win_subword_len=10,
      ctx_subword_len=5
  )
  assert counts == (2, 3)
