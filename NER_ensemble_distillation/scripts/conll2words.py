#! /usr/bin/env python

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

""" Write all tokens appearing in a CoNLL file to a file """

import argparse as ap

from griffin.dataset import Dataset

if __name__ == "__main__":
  p = ap.ArgumentParser()
  p.add_argument('--input-path', required=True)
  p.add_argument('--output-path', required=True)
  p.add_argument('--max-sentence-len', type=int, default=0)
  args = p.parse_args()
  ds = Dataset(args.input_path, args.max_sentence_len)
  with open(args.output_path, 'w', encoding="utf-8") as outfile:
    for sentence in ds.sentences:
      for word in sentence.words:
        outfile.write(word + "\n")
