# -*- coding: utf-8 -*-
"""
Created on Sun May 19 20:08:30 2019

@author: wmy
"""

from parsers import MidiParser
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

parser = MidiParser()

def count(fp="midi", sp="infos/count.txt"):
    files = glob.glob('{}/*.mid*'.format(fp))
    length = 0
    add = 0
    for i, file in enumerate(files):
        print("counting midi file {}/{}...".format(i+1, len(files)))
        sequence = parser.parse(file)
        length += sequence.shape[0]
        add = np.add(np.sum(sequence, axis=0), add)
        pass
    result = np.divide(add, length)
    with open(sp, 'w') as f:
        for i in range(result.shape[0]):
            f.write(str(result[i]) + '\n')
            pass
        pass
    pass

if __name__ == '__main__':
    count()