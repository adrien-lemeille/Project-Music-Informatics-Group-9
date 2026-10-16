
import json
from pathlib import Path
import numpy as np
import pandas as pd


def contains_change(sigs):

    if len(sigs) < 2:
        return False

    initial = sigs[0]
    no_change = True

    k = 1

    while no_change and k < len(sigs):
        no_change = no_change and (sigs[k] == initial)
        k += 1

    return not no_change


def get_uncommon_dataframe():

    """
    converts parsed_beats into a pandas data frame
    """

    PROJECT_DIR = Path(__file__).resolve().parent.parent

    UNCOMMON = PROJECT_DIR / "Data/uncommon"

    rows = []
    for f in sorted((UNCOMMON / "parsed_beats").glob("*.json")):

        d = json.loads(f.read_text(encoding="utf-8"))

        # bars
        ms = d["measures"]

        if not ms:
            continue

        # time signature of each bar (num, den)
        sigs = [(m["time_sig_num"], m["time_sig_den"]) for m in ms]

        # tempo of each bar
        tempos = [m["tempo_bpm"] for m in ms]

        rows.append(dict(
            # id
            track_id=d["youtube_id"],
            # identify the dataset
            dataset="uncommon_meter",
            # path of the file
            audio_path=str(UNCOMMON / "audio" / f"{d['youtube_id']}.wav"),
            # downbeat times
            downbeat_times=[m["downbeat_sec"] for m in ms],
            # beats per bar
            beats_per_bar=[m["time_sig_num"] for m in ms],
            # tempos in BPM
            tempo_per_bar=tempos,

            contains_meter_changes=contains_change(sigs),
            contains_tempo_changes=contains_change(tempos),
        ))

    df_uncommon = pd.DataFrame(rows)

    return df_uncommon


def beats_per_bar(pos):

    N = len(pos)

    if N == 0:
        return []

    beats_per_bar = []
    pos_prev = pos[0]

    for k in range(1, N):

        if pos[k] == 1:
            beats_per_bar.append(pos_prev)

        pos_prev = pos[k]

    # Add the last measure
    beats_per_bar.append(pos_prev)

    return beats_per_bar


def tempo_per_bar(downbeat_times, beats_per_bar):

    tempo_per_bar = []

    # Estimate tempo from consecutive downbeats
    for k in range(len(downbeat_times) - 1):
        length_bar = downbeat_times[k + 1] - downbeat_times[k]

        if length_bar <= 0 or beats_per_bar[k] <= 0:
            tempo_per_bar.append(np.nan)
        else:
            tempo_per_bar.append(
                round(60 * beats_per_bar[k] / length_bar)
            )

    # The last bar has no following downbeat to estimate its tempo
    if len(downbeat_times) > 0:
        if tempo_per_bar:
            tempo_per_bar.append(tempo_per_bar[-1])
        else:
            tempo_per_bar.append(np.nan)

    return tempo_per_bar


def get_harmonix_dataframe():

    PROJECT_DIR = Path(__file__).resolve().parent.parent
    HARMONIX = PROJECT_DIR / "Data/harmonix"

    rows = []
    for f in sorted((HARMONIX / "beats_and_downbeats").glob("*.txt")):

        tid = f.stem

        # each line: beat_time, position_in_bar, bar_number
        data = [l.split() for l in f.read_text().splitlines() if l.strip()]

        if not data:
            continue

        t = np.array([float(r[0]) for r in data])
        pos = np.array([int(float(r[1])) for r in data])

        downbeat_times = t[np.where(pos == 1)]

        beats = beats_per_bar(pos)
        tempos = tempo_per_bar(downbeat_times, beats)

        rows.append(dict(
            # id
            track_id=tid,
            # identify the dataset
            dataset="harmonix",
            # path of the file
            audio_path=str(HARMONIX / "audio" / f"{tid}.wav"),
            # downbeat times
            downbeat_times=downbeat_times.tolist(),
            # number of beats per bar
            beats_per_bar= [int(b) for b in beats],
            # tempos in BPM
            tempo_per_bar=tempos,

            contains_meter_changes=contains_change(beats),
            contains_tempo_changes=contains_change(tempos),
        ))

    df_uharmonix = pd.DataFrame(rows)

    return df_uharmonix

