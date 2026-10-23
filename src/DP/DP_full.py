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
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from scripts.get_data_frame import get_uncommon_dataframe,get_harmonix_dataframe
from src.evaluate import eval_downbeats

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
    bpm = float(librosa.feature.tempo(onset_envelope=delta, sr=Fs, hop_length=hop)[0])*2

    period = (Fs/hop) * 60/bpm
    
    beat_index = dp_beats(delta , period , lam = 100.0)
    
    #For this approximation we will take the amplitude of the beats in order to estimate the down beats (+- 2 frames)
    amp_beats = np.array([ delta[max(0, b - 2): min(b + 3 , len(delta))].max() for b in beat_index])
    
    return(beat_index*hop/Fs, amp_beats)

def dp_downbeat_estimation ( y, Fs , hop=256 , lam = 100.0, candidtes = [3,4,5,6,7,8,9,11,13] ):
    
    beat_index, amp_beats = dp_beat_estimation ( y=y, Fs=Fs , hop=hop , lam=lam )
    
    amp_beats = (amp_beats - amp_beats.mean()) / (amp_beats.std() + 1e-12)
    
    n=len(beat_index)
    
    best_score = -np.inf
    best_k = None
    best_p = None
    db_best = np.array([])
    
    #enumerate all possible beats per bar possible candidates
    for k in candidtes :
        
        
        for phase in range(k):
            
            
            mask = np.zeros(n, bool); mask[phase::k] = True
            n1, n2 = mask.sum(), (~mask).sum()
            down = amp_beats[phase::k]
            other =amp_beats[~mask]
            score = (down.mean() - other.mean()) / np.sqrt(1.0 / n1 + 1.0 / n2)
            
            #print(k,phase,score)
            
            score = np.mean(down)
            
            #print(k,phase,score)
            """
            # Candidate downbeat amplitudes
            down = amp_beats[phase::k]

            if len(down) < 2:
                continue

            # Mean accent strength
            intensity = np.mean(down)

            # Regularity of accents across measures
            variability = np.mean(np.abs(down - intensity))

            # S5 score
            reg=0.5
            score = intensity - reg * variability
            """
            
            
            
            
            print(k,phase,score)
            
            if score>best_score:
                
                best_score = score
                best_p = phase
                best_k = k
                db_best = beat_index[phase::k]
                
    return db_best, best_k, best_p, best_score
            
            








track_id = 'NNMR5MCpYGU'
file_id = track_id+".wav"



t_begin = 30

t_end = 40

df_uncommon = get_uncommon_dataframe()
df_harmonix = get_harmonix_dataframe()
df_merged = pd.concat([df_uncommon, df_harmonix], ignore_index=True, sort=False)

db_times_track=df_merged.loc[df_merged['track_id']==track_id,'downbeat_times']
db_times_track = np.array(db_times_track.iloc[0])

db_truncated = db_times_track[(db_times_track >= t_begin) & (db_times_track < t_end)]-t_begin



PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DATA = PROJECT_DIR / "Data"



file_name = DATA / "uncommon" / "audio" / file_id

track = pydub.AudioSegment.from_file(file_name, format="wav")
track = track.set_channels(1)

# Sample rate
Fs = track.frame_rate

# Raw audio data
track_array = np.array(track.get_array_of_samples()) * 1.0

# Extract the interval from 10 to 11 seconds
n_start = t_begin * Fs
n_end = t_end * Fs
track_array = track_array[n_start:n_end]

#normalize
track_array /= np.max(np.abs(track_array),axis=0)


# Detect onsets
onset = np.array(librosa.onset.onset_detect(
    y=track_array,
    sr=Fs,
    hop_length=256,
    units='frames'
))*1.0


# Novelety function
novelety =  librosa.onset.onset_strength(
    y=track_array,
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

dp_beats_est,amps = dp_beat_estimation( y=track_array, Fs=Fs , hop=256 , lam = 1.0 )


print(dp_beat_estimation)
print(db_truncated)

# Plot
T = np.arange(len(track_array)) / Fs


#%%
plt.title("Track : "+track_id)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.plot(T, track_array)
#plt.vlines(onset_times, ymin=-1, ymax=1, colors='red')
plt.grid()
plt.show()

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.plot(T_novelty,novelety)
plt.vlines(dp_beats_est, ymin=0, ymax=10, colors='red', alpha=0.6,linestyles='--')
plt.grid()
plt.show()

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.vlines(onset , ymin=0, ymax=10, colors='red')
plt.grid()
plt.show()
        
        
dp_downbeats_est, best_k, best_p, best_score = dp_downbeat_estimation ( y=track_array, Fs=Fs , lam = 10, candidtes = [3,4,5,6,8,9,11,13] )
print(dp_downbeats_est, best_k, best_p, best_score)

b,a = dp_beat_estimation ( track_array, Fs , hop=256 , lam = 100.0 )

plt.title("comparaison of downbeats and estimates (dynamic programming)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.plot(T, track_array)
#plt.vlines(b, ymin=-1, ymax=1, colors='red', alpha = 0.6)
plt.vlines(db_truncated, ymin=-1, ymax=1, colors='orange')
plt.grid()
plt.show()

F, cmlc, cmlt, amlc, amlt = eval_downbeats(db_truncated,dp_downbeats_est)
print(F,cmlc, cmlt, amlc, amlt)
#%%
import numpy as np
import matplotlib.pyplot as plt

# Clean scientific style
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, axes = plt.subplots(
    3, 1,
    figsize=(14, 8),
    sharex=True,
    gridspec_kw={"height_ratios": [1.2, 1.5, 0.7]},
    constrained_layout=True
)




axes[0].plot(
    T, track_array,
    
    linewidth=0.65,
    alpha=0.85
)

axes[0].set_title("Audio Waveform", loc="left", fontweight="bold")
axes[0].set_ylabel("Amplitude")
axes[0].grid(alpha=0.2)




axes[1].plot(
    T_novelty, novelety,

    linewidth=1.2,
    label="Novelty function"
)

axes[1].vlines(
    onset_times,
    ymin=0,
    ymax=np.max(novelety),
    colors="orange",
    linestyles="--",
    linewidth=1,
    alpha=0.7,
    label="Detected onsets"
)

axes[1].set_title(
    "Novelty Function & Onset Detection",
    loc="left",
    fontweight="bold"
)
axes[1].set_ylabel("Novelty")
axes[1].legend(frameon=False, loc="upper right")
axes[1].grid(alpha=0.2)




axes[2].vlines(
    dp_beats_est,
    
    ymin=0,
    ymax=1,
    color="green",
    linewidth=1.4,
    alpha=0.85,
    label="Estimated beats"
)

axes[2].set_title("Beat Tracking", loc="left", fontweight="bold")

axes[2].set_yticks([])
axes[2].set_ylim(0, 1.15)
axes[2].set_xlabel("Time (s)")
axes[2].legend(frameon=False, loc="upper right")
axes[2].grid(axis="x", alpha=0.2)


# --------------------------------------------------
# Shared x-axis
# --------------------------------------------------

axes[2].set_xlim(np.min(T), np.max(T))

# Optional: zoom into a specific interval
# axes[2].set_xlim(20, 35)

plt.savefig(
    f"rhythmic_analysis_{track_id}.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
