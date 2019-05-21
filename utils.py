# -*- coding: utf-8 -*-
"""
Created on Sat May 18 23:15:02 2019

@author: wmy
"""

import midi
import numpy as np
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from parsers import MidiParser

class DataLoader(object):
    '''Data Loader'''
    
    def __init__(self, dataset='./midi', name=None):
        self.parser = MidiParser()
        self.__dataset = dataset
        self.__cache = None
        self.name = name
        pass
    
    @property
    def dataset(self):
        return self.__dataset
    
    @dataset.setter
    def dataset(self, value):
        self.__dataset = value
        self.clear()
        pass
    
    def clear(self):
        self.__cache = None
        pass
    
    @property
    def cache(self):
        return self.__cache
    
    def search(self):
        files = glob.glob('{}/*.mid*'.format(self.dataset))
        files = sorted(files)
        return files
    
    def pad(self, sequence, step):
        return np.pad(sequence, ((step, 0), (0, 0)), mode='constant')
    
    def batches(self, step=128, batch_size=16, complete_batch_only=False):
        files = self.search()
        sequences = []
        indexs = []
        if self.__cache == None:
            for i, file in enumerate(files):
                print("loading midi {}/{} ... please wait.".format(i+1, len(files)))
                sequence = self.parser.parse(file)
                sequence = self.pad(sequence, step)
                length = sequence.shape[0]
                sequences.append(sequence)
                for index in range(length-step-1):
                    indexs.append((i, index))
                    pass
                pass
            self.__cache = (sequences, indexs)
            pass
        else:
            sequences, indexs = self.__cache
            pass
        num_data = len(indexs)
        n_complete_batches  = int(num_data/batch_size)
        self.n_batches = int(num_data/batch_size)
        have_res_batch = (num_data/batch_size) > n_complete_batches
        if have_res_batch and complete_batch_only==False:
            self.n_batches += 1
            pass
        np.random.shuffle(indexs)
        for i in range(n_complete_batches):
            batch = indexs[i*batch_size:(i+1)*batch_size]
            X, Y = [], []
            for index in batch:
                i, j = index
                x = sequences[i][j:j+step]
                y = sequences[i][j+step]
                X.append(x)
                Y.append(y)
                pass
            X = np.array(X)
            Y = np.array(Y)
            yield X, Y
        if self.n_batches > n_complete_batches:
            batch = indexs[n_complete_batches*batch_size:]
            X, Y = [], []
            for index in batch:
                i, j = index
                x = sequences[i][j:j+step]
                y = sequences[i][j+step]
                X.append(x)
                Y.append(y)
                pass
            X = np.array(X)
            Y = np.array(Y)
            yield X, Y
        pass
    
    pass

