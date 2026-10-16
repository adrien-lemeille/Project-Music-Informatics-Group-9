# -*- coding: utf-8 -*-

from scripts.get_data_frame import get_uncommon_dataframe
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

import pydub

import csv, io, urllib.request, zipfile

t_begin = 0

t_end = 50

df_uncommon = get_uncommon_dataframe()

db_times_fifteen_steps=df_uncommon.loc[df_uncommon['track_id']=='xpqk9MD6vLM','downbeat_times']
db_times_fifteen_steps = np.array(db_times_fifteen_steps.iloc[0])

db_10_20 = db_times_fifteen_steps[(db_times_fifteen_steps >= t_begin) & (db_times_fifteen_steps < t_end)]-t_begin


# The following makes the plot look nice
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 10),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)




PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA = PROJECT_DIR / "Data"

file_name = DATA / "uncommon" / "audio" / "xpqk9MD6vLM.wav"

fifteen_steps = pydub.AudioSegment.from_file(file_name, format="wav")
fifteen_steps = fifteen_steps.set_channels(1)

# Sample rate
Fs = fifteen_steps.frame_rate
print(Fs)

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

# Plot
T = np.arange(len(fifteen_steps_array)) / Fs


plt.subplot(2, 1, 1)
plt.title("Fifteen Steps - Radiohead")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.plot(T, fifteen_steps_array)
plt.vlines(onset_times, ymin=-1, ymax=1, colors='red')
plt.grid()

plt.subplot(2, 1, 2)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.plot(T_novelty,novelety)
plt.vlines(onset_times, ymin=0, ymax=10, colors='red')
plt.vlines(db_10_20, ymin=0, ymax=10, colors='orange')
plt.grid()
plt.show()


def find_closest_onset (L_onset, L_downbeats):
    
    """
    the down beats and onsets must be ordered from smallest to largest
    returns the closest onset to the down beats and the error
    
    """
    n = len(L_downbeats)
    
    L_closest_onset = np.ones (n)*L_onset[0]
    err_best = np.abs(L_closest_onset - L_downbeats)
    
    
    for k in range (n):

        
        for i in range (len(L_onset)):
            err = np.abs(L_onset[i]-L_downbeats[k])
            
            if err<err_best[k]:
                L_closest_onset[k] = L_onset[i]
                err_best[k] = err
                
    return (L_closest_onset, err_best)
    
L_closest_onset, err = find_closest_onset (onset_times,db_10_20)

print (err)

print(np.round(err * 1000))          # signed errors in ms
print("median error (ms):", np.median(err) * 1000)






