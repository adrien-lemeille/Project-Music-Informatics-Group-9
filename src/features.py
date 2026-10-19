"""Beat candidates and interpretable audio features."""
import numpy as np
import librosa

from pathlib import Path
import sys
import pandas as pd

from get_data_frame import get_uncommon_dataframe, get_harmonix_dataframe
from evaluate import eval_downbeats

import pydub

import csv, io, urllib.request, zipfile


def dp_beats(delta, period, lam=5.0):
    """
    dynamic programming estimation of beat positions as described in the course

    delta    : novelety function delta
    period : estimated perio between 
    lam    : weight

    6.3.2 Beat Tracking by Dynamic Programming
    """

    N = len(delta)

    delta = delta / (delta.std() + 1e-12)

    # range of possible period
    lo, hi = max(1, int(round(period / 2))), int(round(2 * period))
    lags = np.arange(lo, hi + 1)

    # penalty
    pen = -lam * np.log2(lags / period) ** 2

    D = np.zeros(N)
    P = -1 * np.ones(N)

    for n in range(1, N):
        # all possible predecessors
        prev = n - lags
        ok = prev >= 0

        best, arg = 0.0, -1
        if ok.any():
            # Array of all scores
            sc = D[prev[ok]] + pen[ok]

            # index of the best predecessor
            j = int(sc.argmax())

            # if D(n) != Delta(n)
            if sc[j] > 0:
                best, arg = sc[j], int(prev[ok][j])

        D[n] = delta[n] + best
        P[n] = arg

    # Backtracking
    n = int(np.argmax(D))

    beats = [n]
    while P[n] >= 0:
        n = int(P[n])
        beats.append(n)

    # reverese for the beats to be in ascending order
    beats.reverse()

    return np.array(beats)


def dp_beat_estimation(y, Fs, hop=256, lam=10.0, tempo_factor=2):
    delta = librosa.onset.onset_strength(
        y=y, sr=Fs, hop_length=hop
    )

    bpm = float(librosa.feature.tempo(
        onset_envelope=delta,
        sr=Fs,
        hop_length=hop
    )[0])

    bpm *= tempo_factor

    period = (Fs / hop) * 60 / bpm

    beat_index = dp_beats(delta, period, lam=lam)

    amp_beats = np.array([
        delta[max(0, b - 5):min(b + 6, len(delta))].max()
        for b in beat_index
    ])

    return beat_index * hop / Fs, amp_beats


_CHORD_PROC = None


def _get_chord_processors():
    """madmom processors are heavy to build, so we create them once (lazy import)."""
    global _CHORD_PROC
    if _CHORD_PROC is None:
        from madmom.features.chords import CNNChordFeatureProcessor, CRFChordRecognitionProcessor
        _CHORD_PROC = (CNNChordFeatureProcessor(), CRFChordRecognitionProcessor())
    return _CHORD_PROC


def chord_change_feature(chords, beats, tol_ratio=0.3):
    """
    Chord changes (paper 3.1) at the beat level: 1 if a chord change falls on the beat, else 0.
    Each change instant is attached to its closest beat, if it is within
    tol_ratio * (median beat interval), otherwise ignored.

    chords : madmom structured array (start, end, label)
    beats  : beat times in seconds
    """
    out = np.zeros(len(beats))
    if len(chords) < 2 or len(beats) < 2:
        return out

    starts, labels = chords['start'], chords['label']
    tol = tol_ratio * np.median(np.diff(beats))

    for i in range(1, len(chords)):
        if labels[i] != labels[i - 1]:
            k = np.argmin(np.abs(beats - starts[i]))
            if abs(beats[k] - starts[i]) <= tol:
                out[k] = 1.0
    return out


def harmonic_balance_beats(Stft, frequencies, beats, sr, hop):
    """
    Harmonic balance (paper 3.2) at the beat level.
    Bass band [0-150Hz] and snare band [1400-7500Hz] : energy summed over the band,
    then max over a window of +/- L around each beat (L = 1/10 of the beat interval).
    """
    bass = (Stft[(frequencies >= 0) & (frequencies <= 150)] ** 2).sum(axis=0)
    snare = (Stft[(frequencies >= 1400) & (frequencies <= 7500)] ** 2).sum(axis=0)

    ibi = np.median(np.diff(beats)) if len(beats) > 1 else 0.5
    L = 0.1 * ibi
    fps = sr / hop
    n_fr = len(bass)

    e = np.zeros((len(beats), 2))
    for k, b in enumerate(beats):
        i0 = min(max(int(np.floor((b - L) * fps)), 0), n_fr - 1)
        i1 = max(min(int(np.ceil((b + L) * fps)) + 1, n_fr), i0 + 1)
        e[k] = bass[i0:i1].max(), snare[i0:i1].max()

    eps = 1e-10
    rel = (e + eps) / (np.median(e, axis=0) + eps)
    hb = rel[:, 0] / (rel[:, 0] + rel[:, 1])
    return np.column_stack([np.log(rel), hb])


def extract_features_track(audio_path, sr=22050, hop=256, beat_track='librosa', chords=True, normalize=True):
    """Computes features of a song


    audio_path : path WAV 
    sr : fréquence cible en Hz 
    hop : for STFT
    beat_track : type of dp beat tracking (homemade or librosa)
    chords : add the chord change feature (needs madmom)
    normalize : z-score of the frame features on the track


    dict : beat_times (T,)
    X (T, D), matrix of features (T number of estimated beats)

    """

    y, sr = librosa.load(str(audio_path), sr=sr, mono=True)
    # type of beat tracking either librosa or our own

    if beat_track == 'librosa':
        _, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop, trim=False, units='time')

    else:
        beats, _ = dp_beat_estimation(y, Fs=sr, hop=hop, lam=10.0)

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

    # somentimes not exactly the same length not sure why
    n = min(len(onset), len(rms), len(centroid), chroma.shape[1])

    # features compiled in each frames
    frames = np.vstack([rms[:n], onset[:n], centroid[:n], chroma[:, :n]]).T

    # per track standardisation, so that features are comparable between songs
    if normalize:
        frames = (frames - frames.mean(axis=0)) / (frames.std(axis=0) + 1e-8)

    # convert to frame times
    frame_times = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=hop)

    # Interpolation, the beats will fall between 2 frames
    sampled = np.column_stack([np.interp(beats, frame_times, frames[:, j]) for j in range(frames.shape[1])])

    # features computed directly at the beat level (harmonic balance, chord changes)
    beat_level = [harmonic_balance_beats(Stft, frequencies, beats, sr, hop)]
    if chords:
        feat_proc, decode_proc = _get_chord_processors()
        chord_seq = decode_proc(feat_proc(str(audio_path)))
        beat_level.append(chord_change_feature(chord_seq, beats)[:, None])
    sampled = np.hstack([sampled] + beat_level)

    # features of the previous beat for each beats (first beat is itself)
    prev = np.vstack([sampled[0], sampled[:-1]])

    # features of the next beat for each beats (first last is itself)
    nxt = np.vstack([sampled[1:], sampled[-1]])

    # difference between a beat and its neighbors
    diff_prev = sampled - prev
    diff_nxt = nxt - sampled

    # Concaténation du contexte et des différences entre beats voisins.
    X = np.hstack([prev, sampled, nxt, diff_prev, diff_nxt])
    return dict(beat_times=beats, X=np.nan_to_num(X).astype('float32'))


def label_beats(beat_times, downbeat_times, tolerance=0.07):
    """
    Associate downbeats to closest beats with a tolerance 70ms
    used for labelling training data

    beat_times are the positions of the beats estimated with : librosa.beat.beat_track or dp_beats

    downbeat_times are the down beats from the dataset
    """

    if tolerance == 0:
        tolerance = np.mean(beat_times[1:] - beat_times[:-1]) / 5

    labels = np.zeros(len(beat_times), dtype=int)

    for d in downbeat_times:
        i = np.argmin(np.abs(beat_times - d))

        if abs(beat_times[i] - d) <= tolerance:
            labels[i] = 1

    return labels


def build_feature_dictionnary(records, sr=22050, dic=None, beat_track='librosa', **kw):
    """
    compiles the features of songs in a dictionnary so that we only do it once
    """
    if dic is None:
        dic = {}
    for r in records:
        key = (r['dataset'], r['track_id'])

        dic[key] = extract_features_track(r['audio_path'], sr=sr, beat_track=beat_track, **kw)

    return dic