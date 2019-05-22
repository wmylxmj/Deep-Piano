# -*- coding: utf-8 -*-
"""
Created on Sun May 19 13:02:59 2019

@author: wmy
"""

import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
from utils import DataLoader
from model import DeepPianoModel
from optimizer import AdamWithWeightsNormalization
from parsers import MidiParser
from tqdm import tqdm

step = 128

model = DeepPianoModel(step=step)
model.load_weights("weights/deep-piano-128-weights.h5")

parser = MidiParser()

def predict(midi="midi/0001.mid", \
            offset=0, \
            length=256, \
            sp="outputs/generate.mid", \
            tickscale=24):
    sequence = parser.parse(midi)
    # give some conditional melody
    start = sequence[offset:step+offset, :]
    generate = []
    for i in tqdm(range(length)):
        x = np.array([start])
        y = model.predict(x)
        y = np.round(y)
        generate.append(y[0])
        start = np.concatenate([start[1:], y], axis=0)
        pass
    generate = np.array(generate)
    parser.unparse(generate, sp, tickscale=tickscale)
    pass

# predict
predict(midi="midi/0018.mid", offset=960, length=256, sp="outputs/generate.mid", tickscale=24)
      
