#! /usr/bin/env python

import os
import sys
import re
import numpy as np
from sklearn.metrics import precision_recall_curve, brier_score_loss
from sklearn.calibration import calibration_curve

def calib_proc(path):
    with open(path, 'r') as fh:
        labels = []
        confs = []
        line = fh.readline()
        types = re.split(',|:', line.rsplit()[-1])[::2]
        type_map = {}
        for i in range(len(types)):
            type_map[types[i]] = i
        while line:
            line = line.rsplit()
            if line != []:
                temp = [0 for i in range(len(types))]
                label = line[1].split('-')[-1]
                if label != 'O':
                    temp[type_map[label]] = 1
                labels.append(temp)
                confs.append([float(x) for x in re.split(',|:', line[-1])[1::2]])
            line = fh.readline()
        labels = np.asarray(labels)
        confs = np.asarray(confs)
        return np.asarray(labels), np.asarray(confs)

def top_confs(gold, pred):
    pairs = list(zip(pred, gold))
    pairs.sort()
    pairs.reverse()
    N = 2 * sum(gold)
    t_gold = np.zeros(N)
    t_pred = np.zeros(N)
    for i, pair in enumerate(pairs[:N]):
        t_pred[i] = pair[0]
        t_gold[i] = pair[1]
    return t_gold, t_pred

def binned(gold, pred, n_bins):
    bin_size = int(len(gold) / n_bins)
    fop = np.zeros(n_bins)
    mpv = np.zeros(n_bins)
    for i in range(n_bins):
        fop[i] = np.mean(gold[i*bin_size:(i+1)*bin_size])
        mpv[i] = np.mean(pred[i*bin_size:(i+1)*bin_size])
    return fop, mpv

def tace_brier(labels, confs):
    fop, mpv = binned(labels, confs, 20)
    tace = np.mean(np.abs(fop - mpv))
    brier = brier_score_loss(labels, confs)
    return tace, brier

def stratified_brier(labels, confs):
    bs_pos = 0.
    bs_neg = 0.
    n_pos = 0.
    n_neg = 0.
    for label, conf in zip(labels.flatten(), confs.flatten()):
        if label == 1:
            bs_pos += (label-conf)**2
            n_pos += 1
        if label == 0:
            bs_neg += (label-conf)**2
            n_neg += 1
    return bs_pos / n_pos, bs_neg / n_neg

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Error: you must choose a model type and a split")
        print(f"e.g.: python sys.argv[0] crf test")
        exit()
    model = sys.argv[1]
    split = sys.argv[2]
    exp_dir = os.environ.get("GRIFFIN_EXP_DIR") + "/conll_de"

    metrics_list = {}

    for j in range(10):
        pred_file = exp_dir + f"/{model}_{j}/{split}_confidences.txt"
        if not os.path.exists(pred_file):
            continue
        labels, confs = calib_proc(pred_file)
        b_plus, b_minus = stratified_brier(labels, confs)
        total_tace = 0
        total_brier = 0
        examples = 0
        for i in range(labels.shape[-1]):
            t_labels, t_confs = top_confs(labels[:, i], confs[:, i])
            tace, brier = tace_brier(t_labels, t_confs)
            total_tace += tace*len(t_labels)
            total_brier += brier*len(t_labels)
            examples += len(t_labels)
        metrics_list[f"Single {model} {j}"] = [b_plus, b_minus, total_brier/examples, total_tace/examples]

    mean = np.mean(list(metrics_list.values()), 0)
    std_err = np.std(list(metrics_list.values()), 0) / np.sqrt(len(metrics_list))
    metrics_list["Mean of baselines"] = mean
    metrics_list["Std. err. of baselines"] = std_err

    for j in range(10):
        pred_file = exp_dir + f"/{model}_distilled_{j}/{split}_confidences.txt"
        if not os.path.exists(pred_file):
            continue
        labels, confs = calib_proc(pred_file)
        b_plus, b_minus = stratified_brier(labels, confs)
        total_tace = 0
        total_brier = 0
        examples = 0
        for i in range(labels.shape[-1]):
            t_labels, t_confs = top_confs(labels[:, i], confs[:, i])
            tace, brier = tace_brier(t_labels, t_confs)
            total_tace += tace*len(t_labels)
            total_brier += brier*len(t_labels)
            examples += len(t_labels)
        metrics_list[f"Distilled {model} {j}"] = [b_plus, b_minus, total_brier/examples, total_tace/examples]

    print(f"{'Model':<30} {'BS+':<10} {'BS-':<10} {'B-BS':<10} {'B-ECE':<10}")
    for k, v in metrics_list.items():
        print(f"{k + ':':<30} {v[0]*100:<10.2f} {v[1]*100:<10.2f} {v[2]*100:<10.2f} {v[3]*100:<10.2f}")
