#!/usr/bin/env python

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

""" Construct a label dictionary and write to a Pickle file """

import argparse as ap
import pickle
from pathlib import Path
from typing import Dict

from ner.dataset import dataset_from_file
from ner.dataset import Sentence


def build_label_map(input_paths, output_path, max_sentence_len=0, other_tag='O',
                    extra_tag=None):
  """
  Builds the label map
  """

  if not input_paths:
    raise ValueError('Input paths must be specified')

  input_paths = input_paths.split(',')

  label_map: Dict[str, int] = {other_tag: 0}
  if extra_tag:
    label_map['B-' + extra_tag] = len(label_map)
    label_map['I-' + extra_tag] = len(label_map)
  other_count = 0

  # Sometimes we need to grab all of the labels from train, dev, and test
  for path in input_paths:
    ds = dataset_from_file(path, max_sentence_len)
    for sentence in ds.sentence_iter:
      other_count += _labels_for_sentence(sentence, label_map, other_tag)

  if other_count < 1:
    raise ValueError(f"ALERT! No instances of the OUTSIDE ('{other_tag}') tag; "
                     "something's wrong")

  print("Label map:")
  for k, v in label_map.items():
    print(f"{k} {v}")

  # create the output dir(s) if necessary
  Path(output_path).parent.mkdir(parents=True, exist_ok=True)

  print(f"Writing label map to {output_path}...")
  with open(output_path, 'wb') as handle:
    pickle.dump(label_map, handle)

  print(f"Writing plain text version to {output_path}.txt...")
  with open(output_path + ".txt", 'w') as handle:
    for k, v in label_map.items():
      handle.write(f"{k} {v}\n")


def _labels_for_sentence(sentence: Sentence, label_map: Dict[str, int],
                         other_tag: str) -> int:
  """Adds distinct labels from the given sentence to the given label_map
  :returns count of other_tag (usually 'O') observed in the given sentence
  """
  other_count = 0
  for label in sentence.tags or []:
    if label == other_tag:
      other_count += 1
    if label not in label_map:
      label_map[label] = len(label_map)
  return other_count


if __name__ == "__main__":
  p = ap.ArgumentParser()
  p.add_argument('--input-paths',
                 type=str,
                 required=True,
                 help="A comma separated list of paths.")
  p.add_argument('--output-path', required=True)
  p.add_argument('--max-sentence-len', type=int, default=0)
  p.add_argument('--other-tag', default='O')
  p.add_argument('--extra-tag', default=None)
  args = p.parse_args()
  build_label_map(**vars(args))
