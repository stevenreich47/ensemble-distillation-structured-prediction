#! /usr/bin/env python

import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

def create_confs(path):
    with open(path, 'r') as fh:
        spot_confs = []
        sentence_confs = []
        while True:
            confs = fh.readline().rsplit()
            if confs == []:
                if sentence_confs == []:
                    break
                sentence_confs = np.asarray(sentence_confs)
                sentence_confs = np.amax(sentence_confs, axis=0)
                spot_confs.append(sentence_confs)
                sentence_confs = []
            else:
                sentence_confs.append([float(x) for x in re.split(',|:', confs[-1])[1::2]])
        spot_confs = np.asarray(spot_confs)
        np.save("confs.npy", spot_confs)

def create_labels(path):
    with open(path, 'r') as fh:
        spot_labels = []
        sentence_labels = []
        while True:
            labels = fh.readline().rsplit()
            if labels == []:
                if sentence_labels == []:
                    break
                sentence_labels - np.asarray(sentence_labels)
                sentence_labels = np.amax(sentence_labels, axis=0)
                spot_labels.append(sentence_labels)
                sentence_labels = []
            else:
                temp = [0, 0, 0, 0]
                label = labels[1].split('-')[-1]
                if label == 'LOC':
                    temp[0] = 1
                if label == 'ORG':
                    temp[1] = 1
                if label == 'GPE':
                    temp[2] = 1
                if label == 'PER':
                    temp[3] = 1
                sentence_labels.append(temp)
        spot_labels = np.asarray(spot_labels)
        np.save("labels.npy", spot_labels)

def curves(path1, path2, path3, path4):
    confs1 = np.load(path1)
    labels1 = np.load(path2)
    confs2 = np.load(path3)
    labels2 = np.load(path4)
    for i in range(confs1.shape[-1]):
        precision1, recall1, _ = precision_recall_curve(labels1[:, i], confs1[:, i])
        precision2, recall2, _ = precision_recall_curve(labels2[:, i], confs2[:, i])
        fig = plt.figure()
        plt.step(recall1, precision1)
        plt.step(recall2, precision2)
        plt.legend(["NER", "SPOT"])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        fig.savefig(f"pr_plot_{i}", bbox_inches='tight')


if __name__ == "__main__":
    curves("zh_with_alignment/confs.npy",
           "zh_with_alignment/labels.npy",
           "spotting/chinese/confs.npy",
           "spotting/chinese/labels.npy")
