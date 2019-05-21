# -*- coding: utf-8 -*-
"""
Created on Wed May 15 14:49:31 2019

@author: wmy
"""

import numpy as np
import keras.backend as K
from utils import DataLoader
from model import DeepPianoModel
from optimizer import AdamWithWeightsNormalization

def get_rate(count_path="infos/count.txt", threshold=0.01, epsilon=1e-7):
    with open(count_path, 'r') as f:
        count = f.readlines()
        p = []
        for c in count:
            p.append(float(c.strip()))
            pass
        p = np.array(p)
        n = 1 - p
        p = p + epsilon
        n = n + epsilon
        rate = p / n
        for i in range(rate.shape[0]):
            if rate[i] < threshold:
                rate[i] = threshold
                pass
            pass
        pass
    return rate

rate = get_rate()

def loss(y_true, y_pred):
    positive_loss = -(y_true * K.log(y_pred + K.constant(1e-7)))
    negative_loss = -((1-y_true) * K.log((1-y_pred) + K.constant(1e-7)))
    all_loss = positive_loss + rate*negative_loss
    return K.mean(K.sum(all_loss, axis=-1), axis=-1)

def accuracy(y_true, y_pred):
    equal = K.equal(K.mean(K.equal(y_true, K.round(y_pred)), axis=-1), 1)
    return K.mean(equal, axis=-1)

# settings
step = 128
batch_size = 64
pretrained_weights = None

model = DeepPianoModel(step=step)
model.compile(optimizer=AdamWithWeightsNormalization(lr=0.001), loss=loss, metrics=[accuracy])
print("[OK] model created.")

data_loader = DataLoader()

if pretrained_weights != None:
    model.load_weights(pretrained_weights)
    print("[OK] weights loaded.")
    pass

epoches = 10000
for epoch in range(epoches):
    for batch_i, (X, Y) in enumerate(data_loader.batches(step=step, batch_size=batch_size)):
        temp_loss, temp_accuracy = model.train_on_batch(X, Y)
        print("[epoch: {}/{}][batch: {}/{}][loss: {}][accuracy: {}]".format(epoch+1, epoches, \
              batch_i+1, data_loader.n_batches, temp_loss, temp_accuracy))
        if (batch_i+1) % 100 == 0:
            model.save_weights("weights/deep-piano-{}-weights.h5".format(step))
            print("[OK] weights saved.")
            pass
        pass
    model.save_weights("weights/deep-piano-{}-weights.h5".format(step))
    print("[OK] weights saved.")
    pass

