# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:10:39 2026

@author: Adrien Lemeille

"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import sys

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from scripts.get_data_frame import get_uncommon_dataframe

import pydub

import csv, io, urllib.request, zipfile

def dp_beats(delta , period , lam = 100.0):
    """
    novelety    : novelety function delta
    period : estimated perio between 
    lam    : weight

    6.3.2 Beat Tracking by Dynamic Programming
    """
    
    N = len(delta)
    
    delta = delta / (delta.std() + 1e-12)
    
    #range of possible period
    lo, hi = max(1, int(round(period / 2))), int(round(2 * period))
    lags = np.arange(lo, hi + 1)
    
    #penalty
    pen = -lam * np.log2(lags / period) ** 2
    
    D = np.zeros(N)
    P = -1*np.ones(N)
    
    for n in range (1,N):
        #all possible predecessors
        prev = n - lags
        ok = prev >= 0
        
        best, arg = 0.0, -1
        if ok.any():
            #Array of all scores
            sc = D[prev[ok]] + pen[ok]
            
            #index of the best predecessor
            j = int(sc.argmax())
            
            # if D(n) != Delta(n)
            if sc[j] > 0:
                best, arg = sc[j], int(prev[ok][j])
                
        D[n] = delta[n] + best
        P[n] = arg
        
    #Backtracking
    n = int(np.argmax(D))

    beats = [n]
    while P[n] >= 0:
        n = int(P[n])
        beats.append(n)
    
    #reverese for the beats to be in ascending order
    beats.reverse()
    
        
        
    return np.array(beats)


    
def dp_beat_estimation ( y, Fs , hop=256 , lam = 100.0 ):
        
    delta = librosa.onset.onset_strength(y=y, sr=Fs, hop_length=hop)
    
    #initial estimate of the period (we multiply it by 2 because during testing the )
    bpm = 1*float(librosa.feature.tempo(onset_envelope=delta, sr=Fs, hop_length=hop)[0])

    period = (Fs/hop) * 60/bpm
    
    beat_index = dp_beats(delta , period , lam = 100.0)
    
    #For this approximation we will take the amplitude of the beats in order to estimate the down beats (+- 2 frames)
    amp_beats = [ delta[max(0, b - 2): min(b + 3 , len(delta))].max() for b in beat_index]
    
    return(beat_index, amp_beats)









track_id = 'GRMLpJ-v7vk'
file_id = track_id+".wav"


t_begin = 0

t_end = 20

df_uncommon = get_uncommon_dataframe()

db_times_track=df_uncommon.loc[df_uncommon['track_id']==track_id,'downbeat_times']
db_times_track = np.array(db_times_track.iloc[0])

db_truncated = db_times_track[(db_times_track >= t_begin) & (db_times_track < t_end)]-t_begin



PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DATA = PROJECT_DIR / "Data"



file_name = DATA / "uncommon" / "audio" / file_id

fifteen_steps = pydub.AudioSegment.from_file(file_name, format="wav")
fifteen_steps = fifteen_steps.set_channels(1)

# Sample rate
Fs = fifteen_steps.frame_rate

# Raw audio data
fifteen_steps_array = np.array(fifteen_steps.get_array_of_samples()) * 1.0

# Extract the interval from 10 to 11 seconds
n_start = t_begin * Fs
n_end = t_end * Fs
fifteen_steps_array = fifteen_steps_array[n_start:n_end]

#normalize
fifteen_steps_array /= np.max(np.abs(fifteen_steps_array),axis=0)

# Detect onsets
onset = np.array(librosa.onset.onset_detect(
    y=fifteen_steps_array,
    sr=Fs,
    hop_length=256,
    units='frames'
))*1.0


# Novelety function
novelety =  librosa.onset.onset_strength(
    y=fifteen_steps_array,
    sr=Fs,
    hop_length=256,
)


# Convert frame indices to times (seconds)
onset_times = librosa.frames_to_time(onset, sr=Fs, hop_length=256)


# Time vector for the novelty function
T_novelty = librosa.frames_to_time(
    np.arange(len(novelety)),
    sr=Fs,
    hop_length=256
)


dp_beat_estimation = dp_beat_estimation ( y=fifteen_steps_array, Fs=Fs , hop=256 , lam = 100.0 )*256/Fs

print(dp_beat_estimation)
print(db_truncated)

# Plot
T = np.arange(len(fifteen_steps_array)) / Fs


plt.subplot(2, 1, 1)
plt.title("Track : "+track_id)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.plot(T, fifteen_steps_array)
plt.vlines(onset_times, ymin=-1, ymax=1, colors='red')
plt.grid()

plt.subplot(2, 1, 2)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.plot(T_novelty,novelety)
plt.vlines(dp_beat_estimation, ymin=0, ymax=10, colors='red')
plt.vlines(db_truncated, ymin=0, ymax=10, colors='orange')
plt.grid()
plt.show()
        
        
