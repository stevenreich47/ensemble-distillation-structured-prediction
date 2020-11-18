#! /usr/bin/env python

# Copyright 2020 Johns Hopkins University. All Rights Reserved.
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

"""Write predictive distributions from a set of models into a single file"""

import pickle
import re
import os
import sys
import numpy as np

if len(sys.argv) < 2:
  print("Error: You must indicate the type of teacher model: 'crf' or 'iid'")
  print(f"e.g.: python {sys.argv[0]} crf")
  exit()

model = str(sys.argv[1])

exp_dir = os.environ.get('NER_EXP_DIR') + f'/conll_de'
out_dir = exp_dir + f'/{model}_ensemble'

train = open(os.environ.get('NER_DATA_DIR')
             + f'/conll.de.train.iob2', 'r')
teachers = [open(f'{exp_dir}/{model}_{i}/'
                 + 'bio_train_confidences.txt', 'r') for i in range(9)]

if not os.path.isdir(out_dir):
  os.mkdir(out_dir)

with open(f'{out_dir}/teachers.txt', 'w') as fh:
  for line in zip(train, *teachers):
    new_line = [line[0].rstrip()]
    if new_line[0] == '':
      assert line[1].rstrip() == ''
    else:
      new_line.extend([line[i].rsplit()[-1] for i in range(1, len(line))])
    new_line = '\t'.join(new_line) + '\n'
    fh.write(new_line)

train.close()
for i in range(len(teachers)):
  teachers[i].close()

"""
with open("/home/hltcoe/nandrews/exp/steven/experiments/conll_de/label_map.pickle", 'rb') as ph:
    label_map = pickle.load(ph)
rev_label_map = {v: k for k, v in label_map.items()}

for split in ['test']:
    teachers = [open(f"/home/hltcoe/nandrews/exp/steven/experiments/conll_de/{model}/confidences_{i}/bio_{split}_confidences.txt", 'r') for i in range(11, 16)]

    with open(f"{model}_ensemble_{split}_preds.txt", 'w') as fh:
        for line in zip(*teachers):
            new_line = line[0].rsplit()
            if new_line == []:
                assert line[1].rsplit() == []
            else:
                new_line = new_line[:2]
                rolling_sum = np.zeros(len(rev_label_map))
                for i in range(5):
                    rolling_sum += np.asarray([float(x) for x in re.split(':|,', line[-i].rsplit()[-1])[1::2]])
                    new_line.append(rev_label_map[np.argmax(rolling_sum)])

            new_line = '\t'.join(new_line) + '\n'
            fh.write(new_line)

    for teacher in teachers:
        teacher.close()
"""
