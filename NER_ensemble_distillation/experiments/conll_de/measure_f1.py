#! /usr/bin/env python

import sys
import os
from statistics import stdev
from math import sqrt

from ner.conlleval import evaluate, full_metrics

if len(sys.argv) < 3:
    print("Error: you must choose a model type and a split")
    print(f"e.g.: python {sys.argv[0]} crf test")
    exit()

model = str(sys.argv[1])
split = str(sys.argv[2])

f1_list ={}

exp_dir = os.environ.get("NER_EXP_DIR") + "/conll_de"

for i in range(10):
    pred_file = exp_dir + f"/{model}_{i}/{split}_predictions.txt"
    if not os.path.exists(pred_file):
        continue
    with open(pred_file, 'r') as fh:
        predictions = fh.readlines()
        f1 = full_metrics(evaluate(predictions)).overall.fscore
        f1_list[f"Single {model} {i}"] = f1

mean = sum(f1_list.values()) / len(f1_list)
std_err = stdev(f1_list.values()) / sqrt(len(f1_list))

f1_list["Mean of baselines"] = mean
f1_list["Std. Err. of baselines"] = std_err

for i in range(10):
    pred_file = exp_dir + f"/{model}_distilled_{i}/{split}_predictions.txt"
    if not os.path.exists(pred_file):
        continue
    with open(pred_file, 'r') as fh:
        predictions = fh.readlines()
        f1 = full_metrics(evaluate(predictions)).overall.fscore
        f1_list[f"Distilled {model} {i}"] = f1

print(f"{'Model':<30} F1")
for k, v in f1_list.items():
    print(f"{k + ':':<30} {v*100:.2f}")
