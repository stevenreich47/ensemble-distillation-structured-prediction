#! /usr/bin/env python

import os
import sys
import re
import numpy as np
import tensorflow as tf

class Model(object):
    def __init__(self):
        self.temp = tf.Variable(1.0)

    def __call__(self, x):
        return x / self.temp

def loss(labels, logits):
    return tf.reduce_sum(
               tf.nn.softmax_cross_entropy_with_logits(
                   labels, logits
               )
           )

def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as t:
        current_loss = loss(outputs, model(inputs))
    d_temp = t.gradient(current_loss, model.temp)
    model.temp.assign_sub(learning_rate * d_temp)

def logit_proc(path):
    with open(path + f'dev_logits.txt', 'r') as fh:
        labels = []
        logits = []
        line = fh.readline()
        types = re.split(',|:', line.rsplit()[-1])[::2]
        type_map = {}
        for i in range(len(types)):
            type_map[types[i]] = i
        while line:
            line = line.rsplit()
            if line != []:
                temp = [0 for i in range(len(types))]
                label = line[1] #.split('-')[-1]
                #if label != 'O':
                temp[type_map[label]] = 1
                labels.append(temp)
                logits.append([float(x) for x in re.split(',|:', line[-1])[1::2]])
            line = fh.readline()
        labels = np.asarray(labels)
        logits = np.asarray(logits)
        np.save(path + f'dev_labels.npy', labels)
        np.save(path + f'dev_logits.npy', logits)
        return np.asarray(labels), np.asarray(logits)

if __name__ == '__main__':
    path = str(sys.argv[1])
    if not os.path.exists(path + 'dev_logits.npy'):
        labels, logits = logit_proc(path)
    else:
        logits = np.load(path + 'dev_logits.npy')
        labels = np.load(path + 'dev_labels.npy')

    model = Model()

    temps = []
    epochs = range(350)
    for epoch in epochs:
        temps.append(model.temp.numpy())
        current_loss = loss(labels, model(logits))

        train(model, logits, labels, learning_rate=0.00001)
        if epoch > 300:
            print(f"Epoch {epoch}: temp={temps[-1]}, loss={current_loss}")

    print(temps[-1])
