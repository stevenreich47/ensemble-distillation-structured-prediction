#!/usr/bin/env python

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
# ==============================================================================

"""Python version of the evaluation script from CoNLL'00-"""

# Intentional differences:
# - accept any space as delimiter by default
# - optional file argument (default STDIN)
# - option to set boundary (-b argument)
# - LaTeX output (-l argument) not supported
# - raw tags (-r argument) not supported

import re
import sys
from argparse import Namespace
from collections import defaultdict
from typing import Dict, TextIO, Iterable, NamedTuple

ANY_SPACE = '<SPACE>'


class FormatError(Exception):
  pass


class Metrics(NamedTuple):
  tp: int
  fp: int
  fn: int
  prec: float
  rec: float
  fscore: float


class FullMetrics(NamedTuple):
  overall: Metrics
  by_type: Dict[str, Metrics]


class EvalCounts:
  """Output of the evaluation function"""

  def __init__(self):
    self.correct_chunk = 0  # number of correctly identified chunks
    self.correct_tags = 0  # number of correct chunk tags
    self.found_correct = 0  # number of chunks in corpus
    self.found_guessed = 0  # number of identified chunks
    self.token_counter = 0  # token counter (ignores sentence breaks)

    # counts by type
    self.t_correct_chunk: Dict[str, int] = defaultdict(int)
    self.t_found_correct: Dict[str, int] = defaultdict(int)
    self.t_found_guessed: Dict[str, int] = defaultdict(int)


def parse_args(argv) -> Namespace:
  """TODO describe this function"""
  # pylint: disable=import-outside-toplevel
  import argparse
  parser = argparse.ArgumentParser(
      description='evaluate tagging results using CoNLL criteria',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  arg = parser.add_argument
  arg('-b', '--boundary', metavar='STR', default='-X-',
      help='sentence boundary')
  arg('-d', '--delimiter', metavar='CHAR', default=ANY_SPACE,
      help='character delimiting items in input')
  arg('-o', '--otag', metavar='CHAR', default='O',
      help='alternative outside tag')
  arg('file', nargs='?', default=None)
  return parser.parse_args(argv)


def parse_tag(t):
  """TODO describe this function"""
  m = re.match(r'^([^-]*)-(.*)$', t)
  return m.groups() if m else (t, '')


def evaluate(iterable: Iterable[str], options: Namespace = None) -> EvalCounts:
  """TODO describe this function"""
  if options is None:
    options = parse_args([])  # use defaults

  counts = EvalCounts()
  num_features = None  # number of features per line
  in_correct = False  # currently processed chunks is correct until now
  last_correct = 'O'  # previous chunk tag in corpus
  last_correct_type = ''  # type of previously identified chunk tag
  last_guessed = 'O'  # previously identified chunk tag
  last_guessed_type = ''  # type of previous chunk tag in corpus

  for line in iterable:
    line = line.rstrip('\r\n')

    if options.delimiter == ANY_SPACE:
      features = line.split()
    else:
      features = line.split(options.delimiter)

    if num_features is None:
      num_features = len(features)
    elif num_features != len(features) and len(features) != 0:
      raise FormatError(
          f'unexpected number of features: {len(features)} ({num_features})')

    if len(features) == 0 or features[0] == options.boundary:
      features = [options.boundary, 'O', 'O']
    if len(features) < 3:
      raise FormatError('unexpected number of features in line %s' % line)

    guessed, guessed_type = parse_tag(features.pop())
    correct, correct_type = parse_tag(features.pop())
    first_item = features.pop(0)

    if first_item == options.boundary:
      guessed = 'O'

    end_correct = end_of_chunk(last_correct, correct,
                               last_correct_type, correct_type)
    end_guessed = end_of_chunk(last_guessed, guessed,
                               last_guessed_type, guessed_type)
    start_correct = start_of_chunk(last_correct, correct,
                                   last_correct_type, correct_type)
    start_guessed = start_of_chunk(last_guessed, guessed,
                                   last_guessed_type, guessed_type)

    if in_correct:
      if end_correct and end_guessed and last_guessed_type == last_correct_type:
        in_correct = False
        counts.correct_chunk += 1
        counts.t_correct_chunk[last_correct_type] += 1
      elif end_correct != end_guessed or guessed_type != correct_type:
        in_correct = False

    if start_correct and start_guessed and guessed_type == correct_type:
      in_correct = True

    if start_correct:
      counts.found_correct += 1
      counts.t_found_correct[correct_type] += 1
    if start_guessed:
      counts.found_guessed += 1
      counts.t_found_guessed[guessed_type] += 1
    if first_item != options.boundary:
      if correct == guessed and guessed_type == correct_type:
        counts.correct_tags += 1
      counts.token_counter += 1

    last_guessed = guessed
    last_correct = correct
    last_guessed_type = guessed_type
    last_correct_type = correct_type

  if in_correct:
    counts.correct_chunk += 1
    counts.t_correct_chunk[last_correct_type] += 1

  return counts


def uniq(iterable):
  seen = set()
  return [i for i in iterable if not (i in seen or seen.add(i))]


def calculate_metrics(correct: int, guessed: int, total: int) -> Metrics:
  tp, fp, fn = correct, guessed - correct, total - correct
  p = 0 if tp + fp == 0 else 1. * tp / (tp + fp)
  r = 0 if tp + fn == 0 else 1. * tp / (tp + fn)
  f = 0 if p + r == 0 else 2 * p * r / (p + r)
  return Metrics(tp, fp, fn, p, r, f)


def full_metrics(counts: EvalCounts) -> FullMetrics:
  """TODO describe this function"""
  c = counts
  overall: Metrics = calculate_metrics(
      c.correct_chunk, c.found_guessed, c.found_correct)
  by_type: Dict[str, Metrics] = {}
  for t in uniq(list(c.t_found_correct.keys()) +
                list(c.t_found_guessed.keys())):
    by_type[t] = calculate_metrics(
        c.t_correct_chunk[t], c.t_found_guessed[t], c.t_found_correct[t])
  return FullMetrics(overall, by_type)


def report(counts: EvalCounts, out: TextIO = None) -> None:
  """TODO describe this function"""
  if out is None:
    out = sys.stdout

  metrics: FullMetrics = full_metrics(counts)

  out.write(f'processed {counts.token_counter} tokens with '
            f'{counts.found_correct} phrases; ')
  out.write(f'found: {counts.found_guessed} phrases; '
            f'correct: {counts.correct_chunk}.\n')

  if counts.token_counter > 0:
    accuracy = counts.correct_tags / counts.token_counter
    out.write(f'accuracy: {100. * accuracy:6.2f}%; ')
    out.write(f'precision: {100. * metrics.overall.prec:6.2f}%; ')
    out.write(f'recall: {100. * metrics.overall.rec:6.2f}%; ')
    out.write(f'FB1: {100. * metrics.overall.fscore:6.2f}\n')

  for tag, m in sorted(metrics.by_type.items()):
    out.write(f'{tag:>17}: ')
    out.write(f'precision: {100. * m.prec:6.2f}%; ')
    out.write(f'recall: {100. * m.rec:6.2f}%; ')
    found_guessed = counts.t_found_guessed[tag]
    out.write(f'FB1: {100. * m.fscore:6.2f}  {found_guessed :d}\n')


def end_of_chunk(prev_tag, tag, prev_type, type_):
  """check if a chunk ended between the previous and current word
  arguments: previous and current chunk tags, previous and current types
  """
  if prev_tag in ('E', 'S'):
    return True

  if prev_tag in ('B', 'I') and tag in ('B', 'S', 'O'):
    return True

  if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
    return True

  # these chunks are assumed to have length 1
  if prev_tag in (']', '['):
    return True

  return False


def start_of_chunk(prev_tag, tag, prev_type, type_):
  """check if a chunk started between the previous and current word
  arguments: previous and current chunk tags, previous and current types
  """
  if tag in ('B', 'S'):
    return True

  if prev_tag in ('E', 'S', 'O') and tag in ('I', 'E'):
    return True

  if tag != 'O' and tag != '.' and prev_type != type_:
    return True

  # these chunks are assumed to have length 1
  if tag in ('[', ']'):
    return True

  return False


def main(argv):
  args = parse_args(argv[1:])

  if args.file is None:
    counts = evaluate(sys.stdin, args)
  else:
    with open(args.file) as f:
      counts = evaluate(f, args)
  report(counts)


if __name__ == '__main__':
  sys.exit(main(sys.argv))
