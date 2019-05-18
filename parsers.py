# -*- coding: utf-8 -*-
"""
Created on Sat May 18 21:47:44 2019

@author: wmy
"""

import midi
import numpy as np
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

class MidiParser(object):
    '''Midi Parser'''
    
    def __init__(self, name=None):
        self.__lowest_pitch = 21
        self.__highest_pitch = 108
        self.__pitch_span = self.__highest_pitch - self.__lowest_pitch + 1
        self.name = name
        pass
    
    @property
    def lowest_pitch(self):
        return self.__lowest_pitch
    
    @lowest_pitch.setter
    def lowest_pitch(self, pitch):
        if isinstance(pitch, int):
            if pitch >= 0:
                if pitch <= self.__highest_pitch:
                    self.__lowest_pitch = pitch
                    self.__pitch_span = self.__highest_pitch - self.__lowest_pitch + 1
                else:
                    raise ValueError("lowest pitch must be lower than highest pitch")
            else:
                raise ValueError("expected lowest pitch >= 0")
        else:
            raise ValueError("lowest pitch must be the type of int")
    
    @property
    def highest_pitch(self):
        return self.__highest_pitch
    
    @highest_pitch.setter
    def highest_pitch(self, pitch):
        if isinstance(pitch, int):
            if pitch <= 127:
                if pitch >= self.__lowest_pitch:
                    self.__highest_pitch = pitch
                    self.__pitch_span = self.__highest_pitch - self.__lowest_pitch + 1
                else:
                    raise ValueError("highest pitch must be higher than lowest pitch")
            else:
                raise ValueError("expected highest pitch <= 127")
        else:
            raise ValueError("highest pitch must be the type of int")
            
    @property
    def pitch_span(self):
        return self.__pitch_span
    
    def parse(self, fp, tracks=None):
        pattern = midi.read_midifile(fp)
        if tracks != None:
            if not isinstance(tracks, list):
                raise ValueError("tracks must be a list.")
            new_pattern = midi.Pattern()
            new_pattern.resolution = 480
            for index in tracks:
                if not isinstance(index, int):
                    raise ValueError("element in tracks must be int.")
                new_pattern.append(pattern[index])
                pass
            pattern = new_pattern
            pass
        sequence = []
        state = [0 for x in range(self.__pitch_span)]
        sequence.append(state)
        time_left = [track[0].tick for track in pattern]
        posns = [0 for track in pattern]
        time = 0
        while True:
            # duration: 1/64
            if time % (pattern.resolution//16) == (pattern.resolution//32):
                last_state = state
                state = [last_state[x] for x in range(self.__pitch_span)]
                sequence.append(state)
                pass
            for i in range(len(time_left)):
                while time_left[i] == 0:
                    track = pattern[i]
                    pos = posns[i]
                    evt = track[pos]
                    if isinstance(evt, midi.NoteEvent):
                        if (evt.pitch >= self.__lowest_pitch) or (evt.pitch <= self.__highest_pitch):
                            if isinstance(evt, midi.NoteOffEvent) or evt.velocity == 0:
                                state[evt.pitch-self.__lowest_pitch] = 0                      
                            else:
                                state[evt.pitch-self.__lowest_pitch] = 1   
                        pass
                    try:
                        time_left[i] = track[pos + 1].tick
                        posns[i] += 1
                    except IndexError:
                        time_left[i] = None
                    pass
                if time_left[i] is not None:
                    time_left[i] -= 1
            if all(t is None for t in time_left):
                break
            time += 1
            pass
        sequence = np.array(sequence)
        return sequence
    
    def unparse(self, sequence, sp, tickscale=24, velocity=80):
        sequence = np.array(sequence)
        pattern = midi.Pattern()
        pattern.resolution = 480
        track = midi.Track()
        pattern.append(track)
        tickscale = tickscale
        lastcmdtime = 0
        prevstate = [0 for x in range(self.__pitch_span)]
        for time, state in enumerate(sequence):  
            offNotes = []
            onNotes = []
            for i in range(self.__pitch_span):
                n = state[i]
                p = prevstate[i]
                if p == 1 and n == 0:
                    offNotes.append(i)
                elif p == 0 and n == 1:
                    onNotes.append(i)
                pass
            for note in offNotes:
                tick = (time - lastcmdtime) * tickscale
                pitch = note + self.__lowest_pitch
                event = midi.NoteOffEvent(tick=tick, pitch=pitch)
                track.append(event)
                lastcmdtime = time
            for note in onNotes:
                tick = (time - lastcmdtime) * tickscale
                pitch = note + self.__lowest_pitch
                event = midi.NoteOnEvent(tick=tick, velocity=velocity, pitch=pitch)
                track.append(event)
                lastcmdtime = time
                pass
            prevstate = state
            pass
        eot = midi.EndOfTrackEvent(tick=1)
        track.append(eot)
        midi.write_midifile(sp, pattern)
        pass
    
    def plot(self, fp, sp):
        sequence = self.parse(fp)
        plt.rcParams['figure.dpi'] = 300
        plt.imshow(sequence, aspect='auto')
        plt.savefig(sp)
        plt.rcParams['figure.dpi'] = 100
        plt.close()
        pass
    
    pass

