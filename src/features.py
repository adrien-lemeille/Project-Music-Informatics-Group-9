"""Beat candidates and interpretable audio features."""
import numpy as np
import librosa

BASE_NAMES = ['rms', 'onset', 'log_bass', 'bass_flux', 'chroma_nov1']


def dp_beats(delta, period, lam=5.0):
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



def feature_names():
    return (list(BASE_NAMES) + [f'dprev:{n}' for n in BASE_NAMES] + [f'dnext:{n}' for n in BASE_NAMES]  + [f'zloc:{n}' for n in BASE_NAMES])


def _shift(A, k):
    return A[np.clip(np.arange(len(A)) + k, 0, len(A) - 1)]


def _interp(sig, beats, fps):
    return np.interp(beats, np.arange(len(sig)) / fps, sig)


def _sample_max(sig, beats, fps, L):
    n = len(sig)
    out = np.empty(len(beats))
    for k, b in enumerate(beats):
        i0 = min(max(int(np.floor((b - L) * fps)), 0), n - 1)
        i1 = max(min(int(np.ceil((b + L) * fps)) + 1, n), i0 + 1)
        out[k] = sig[i0:i1].max()
    return out


def _beat_sync(F, beats, fps):
    n_fr = F.shape[1]
    idx = np.clip(np.round(beats * fps).astype(int), 0, n_fr - 1)
    step = max(1, int(np.median(np.diff(idx)))) if len(idx) > 1 else 1
    edges = np.append(idx, min(n_fr, idx[-1] + step))
    return np.array([F[:, a:max(b, a + 1)].mean(axis=1) for a, b in zip(edges[:-1], edges[1:])])


def chroma_novelty(chroma_b):
    c = chroma_b / (np.linalg.norm(chroma_b, axis=1, keepdims=True) + 1e-8)
    out = np.zeros(len(c))
    out[1:] = 1 - (c[1:] * c[:-1]).sum(axis=1)
    return out


def _band_mask(frequencies, lo, hi):
    return (frequencies >= lo) & (frequencies <= hi)


def bass_energy_beats(Stft, frequencies, beats, sr, hop, lo=0, hi=150):
    bass = (Stft[_band_mask(frequencies, lo, hi)] ** 2).sum(axis=0)
    ibi = np.median(np.diff(beats)) if len(beats) > 1 else 0.5
    e = _sample_max(bass, beats, sr / hop, 0.1 * ibi)
    eps = 1e-10
    return np.log((e + eps) / (np.median(e) + eps))


def bass_flux(Stft, frequencies, lo=0, hi=150):
    band = np.log1p(Stft[_band_mask(frequencies, lo, hi)])
    d = np.diff(band, axis=1, prepend=band[:, :1])
    return np.maximum(d, 0).sum(axis=0)


def extract_features_track(audio_path, sr=22050, hop=256, beat_track='librosa', normalize=True, compression=True, win=4):
    names = feature_names()
    y, sr = librosa.load(str(audio_path), sr=sr, mono=True)

    if beat_track == 'librosa':
        _, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop, trim=False, units='time')
    else:
        beats, _ = dp_beat_estimation(y, Fs=sr, hop=hop, lam=10.0)
    beats = np.asarray(beats, dtype=float)

    if len(beats) < 2:
        return dict(beat_times=beats, X=np.empty((0, len(names)), dtype='float32'), names=names)

    fps = sr / hop
    Stft = np.abs(librosa.stft(y, hop_length=hop))
    frequencies = librosa.fft_frequencies(sr=sr)
    L = 0.1 * np.median(np.diff(beats))


    mel_db = librosa.power_to_db(librosa.feature.melspectrogram(S=Stft ** 2, sr=sr))
    onset = librosa.onset.onset_strength(S=mel_db, sr=sr, hop_length=hop)
    rms = librosa.feature.rms(S=Stft)[0]
    chroma = librosa.feature.chroma_stft(S=Stft ** 2, sr=sr)
    flux = bass_flux(Stft, frequencies)

    squash = np.log1p if compression else (lambda v: v)


    rms_b = _interp(squash(rms), beats, fps)
    onset_b = squash(_sample_max(onset, beats, fps, L))
    log_bass_b = bass_energy_beats(Stft, frequencies, beats, sr, hop)
    bass_flux_b = squash(_sample_max(flux, beats, fps, L))
    chroma_b = _beat_sync(chroma, beats, fps)

    S = np.column_stack([rms_b, onset_b, log_bass_b, bass_flux_b, chroma_novelty(chroma_b)])

    if normalize:
        S = (S - S.mean(axis=0)) / (S.std(axis=0) + 1e-8)

    diff_prev = S - _shift(S, -1)
    diff_nxt = _shift(S, 1) - S

    stack = np.stack([_shift(S, k) for k in range(-win, win + 1)])
    zloc = (S - stack.mean(axis=0)) / (stack.std(axis=0) + 1e-8)

    X = np.hstack([S, diff_prev, diff_nxt, zloc])
    return dict(beat_times=beats, X=np.nan_to_num(X).astype('float32'), names=names)

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