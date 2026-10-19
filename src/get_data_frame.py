
import json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
HARMONIX = PROJECT_DIR / "Data/harmonix"
UNCOMMON = PROJECT_DIR / "Data/uncommon"



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


def get_uncommon_dataframe(root=UNCOMMON):
    """
    root : folder containing parsed_beats/ and audio/.

    """
    root = Path(root)
    rows = []
    
    for file in sorted((root / 'parsed_beats').glob('*.json')):
        
        data = json.loads(file.read_text(encoding='utf-8'))
        measures = data.get('measures', [])
       
        signatures = [(int(m['time_sig_num']), int(m['time_sig_den'])) for m in measures]
        
        rows.append(dict(track_id=str(data['youtube_id']), dataset='uncommon_meter',
                         audio_path=str(root / 'audio' / (str(data['youtube_id']) + '.wav')),
                         downbeat_times=[float(m['downbeat_sec']) for m in measures],
                         time_signatures=signatures,
                         beats_per_bar=[a for a, _ in signatures],
                         tempo_per_bar=[float(m['tempo_bpm']) for m in measures],
                         contains_meter_changes=contains_change(signatures)))
    # Une ligne du DataFrame correspond à un morceau entier.
    return pd.DataFrame(rows)




def get_harmonix_dataframe(root=HARMONIX):
    """
    root : folder containing parsed_beats/ and audio/.

    """
    root = Path(root)
    rows = []
    for file in sorted((root / 'beats_and_downbeats').glob('*.txt')):
        
        records = [line.split() for line in file.read_text(encoding='utf-8').splitlines() if line.strip()]
        
        # times and position in the measure of the beats
        times = np.array([float(r[0]) for r in records])
        positions = np.array([int(float(r[1])) for r in records])
        
        #to identify down beats
        starts = np.flatnonzero(positions == 1)
        
        if not len(starts):
            continue
        
        # Only completed bars can be measured reliably (last bar has unknown length)
        counts = [int(np.max(positions[a:b])) for a, b in zip(starts[:-1], starts[1:])]
        counts.append(None) #last bar, might delete
        
        #All the ones are down beats
        downbeats = times[starts].tolist()
        
        tempos = [60 * n / (downbeats[i+1] - downbeats[i]) if downbeats[i+1] > downbeats[i]
                  else np.nan for i, n in enumerate(counts[:-1])]
        tempos.append(np.nan)
        rows.append(dict(track_id=file.stem, dataset='harmonix',
                         audio_path=str(root / 'audio' / (file.stem + '.wav')),
                         downbeat_times=downbeats,
                         time_signatures=[None] * len(downbeats),#we know only the beats per bar, not the time signature
                         beats_per_bar=counts, tempo_per_bar=tempos,
                         contains_meter_changes=contains_change(counts[:-1])))
    return pd.DataFrame(rows)


def get_full_dataframe(root_u=UNCOMMON,root_h=HARMONIX, include_harmonix = False):
    
    df = get_uncommon_dataframe(root_u)
    
    if include_harmonix:
        dh = get_harmonix_dataframe(root=root_h)
    
        df = pd.concat([df, dh], ignore_index=True, sort=False)
    
    return df[df.apply(lambda r: Path(r['audio_path']).is_file() , axis=1)].reset_index(drop=True) #We only keep the annotations that have audio files



