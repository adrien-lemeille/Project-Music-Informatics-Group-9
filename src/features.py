
"""Beat candidates and interpretable audio features."""
import numpy as np
import librosa


from pathlib import Path
import sys
import pandas as pd

from scripts.get_data_frame import get_uncommon_dataframe,get_harmonix_dataframe
from src.evaluate import eval_downbeats

import pydub

import csv, io, urllib.request, zipfile

def dp_beats(delta , period , lam = 10.0):
    """
    dynamic programming estimation of beat positions as described in the course
    
    delta    : novelety function delta
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
    


def extract_features_track(audio_path, sr=22050, hop=256, beat_track = 'librosa'):
    """Computes features of a song


    audio_path : path WAV 
    sr : fréquence cible en Hz 
    hop : for STFT
    beat_track : type of dp beat tracking (homemade or librosa)


    dict : beat_times (T,)
    X (T, D), matrix of features (T number of estimated beats)

    """


    y, sr = librosa.load(str(audio_path), sr=sr, mono=True)
    # type of beat tracking either librosa or our own
    
    if beat_track=='librosa' :
        _, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop, trim=False, units='time')
    
    else :
        beats,_ = dp_beat_estimation ( y, Fs=sr , hop=hop , lam = 10.0 )
        
    beats = np.asarray(beats, dtype=float)


    if len(beats) == 0:
        return dict(beat_times=beats, X=np.empty((0, 0)))

    
    # STFT 
    Stft = np.abs(librosa.stft(y, hop_length=hop))
    frequencies = librosa.fft_frequencies(sr=sr)


    # onset strength (for rythmique accent)
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)


    # local energy
    rms = librosa.feature.rms(S=Stft)[0]


    # spectral centroids
    centroid = librosa.feature.spectral_centroid(S=Stft, sr=sr)[0] / sr


    # chroma features (not as good as chord changes bu can still be usefull)
    chroma = librosa.feature.chroma_stft(S=Stft, sr=sr)

    # Energie of the bass
    low = np.mean(Stft[(frequencies >= 30) & (frequencies < 180)]**2, axis=0)


    # Energie of the medius/highs
    high = np.mean(Stft[(frequencies >= 180) & (frequencies < 3000)]**2, axis=0)

    #somentimes not exactly the same length not sure why
    n = min(len(onset), len(rms), len(centroid), len(low), len(high), chroma.shape[1])
    
    #features compiled in each frames
    frames = np.vstack([np.log1p(rms[:n]), np.log1p(onset[:n]), centroid[:n],
                        np.log1p(low[:n]), np.log1p(high[:n]), chroma[:, :n]]).T
    #convert to frame times 
    frame_times = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=hop)


    # Interpolation, the beats will fall between 2 frames
    sampled = np.column_stack([np.interp(beats, frame_times, frames[:, j]) for j in range(frames.shape[1])])
    
    # features of the previous beat for each beats (first beat is itself)
    prev = np.vstack([sampled[0], sampled[:-1]])

    # features of the next beat for each beats (first last is itself)
    nxt = np.vstack([sampled[1:], sampled[-1]])
    
    # difference between a beat and its neighbors
    diff_prev = sampled-prev
    diff_nxt = nxt-sampled

    # Concaténation du contexte et des différences entre beats voisins.
    X = np.hstack([prev, sampled, nxt, diff_prev, diff_nxt])
    return dict(beat_times=beats, X=np.nan_to_num(X).astype('float32'))


def label_beats(beat_times, downbeat_times, tolerance=0.07):
    """
    Associate downbeats to closest beats with a tolerance 70ms
    """
    beat_times = np.asarray(beat_times)
    
    labels = np.zeros(len(beat_times))
    
    
    for t in downbeat_times:
        
        if not len(beat_times):
            break
        
        for i in np.argsort(np.abs(beat_times - t)):
            
            if abs(beat_times[i] - t) > tolerance:
                break
            
            labels[i] = 1
            break
        
    return labels


def build_cache(records, sr=22050,dic = {}):
    """
    compiles the features of songs in a dictionnary so that we only do it once
    """
    for r in records:
        key = (r['dataset'], r['track_id'])
        try:
            dic[key] = extract_features_track(r['audio_path'], sr=sr)
        except Exception as exc:
            print('Skipping', key, ':', exc)
    return dic
