import json
from pathlib import Path
import numpy as np
import pandas as pd



def contains_change(sigs):
    
    initial = sigs[0]
    no_change=True
    
    k=1
    
    while no_change and k<len(sigs):
        no_change =  no_change & (sigs[k]==initial)
        k+=1
        
    return (not no_change)



def get_uncommon_dataframe ():
    
    """
    converts parsed_beats into a panda data frame
    """
    
    
    PROJECT_DIR = Path(__file__).resolve().parent.parent
    
    UNCOMMON = PROJECT_DIR / "Data/uncommon"

    
    rows = []
    for f in sorted((UNCOMMON / "parsed_beats").glob("*.json")):
        
        d = json.loads(f.read_text(encoding="utf-8"))
        
        #bars
        ms = d["measures"]
        
        #time signature of each bar (den, num)
        sigs = [(m["time_sig_num"], m["time_sig_den"]) for m in ms]
        
        # tempo of each bar in
        tempos = [m["tempo_bpm"]  for m in ms]
        
        rows.append(dict(
            #id
            track_id=d["youtube_id"],
            #identify the dataset
            dataset="uncommon_meter",
            #path of the file if needeed (all the audios need to be in the audio folder)
            audio_path=str(UNCOMMON / "audio" / f"{d['youtube_id']}.wav"),
            # Down beat times
            downbeat_times=[m["downbeat_sec"] for m in ms],
            #beats per bar (the numerator of the time signature)
            beats_per_bar = [m["time_sig_num"] for m in ms],
            #tempos in BPM
            tempo_per_bar=tempos ,
            
            contains_meter_changes=contains_change(sigs),
            contains_tempo_changes=contains_change(tempos),
            ))
    df_uncommon = pd.DataFrame(rows)
    
    return df_uncommon



def beats_per_bar (pos):
    
    N= len(pos)
    
    beats_per_bar=[]
    pos_prev = pos[0]
    
    for k in range (1,N):
        
        if k==N-1:
            beats_per_bar.append(pos[k])
            
        elif pos[k]==1:
            beats_per_bar.append(pos_prev)
        
        pos_prev=pos[k]
        
    return(beats_per_bar)

def tempo_per_bar (downbeat_times, beats_per_bar):
    
    k=0
    
    tempo_per_bar=[]
    
    for k in range (len(downbeat_times)-1):
        length_bar = downbeat_times[k+1]-downbeat_times[k]
        tempo_per_bar.append( round (60*beats_per_bar[k]/length_bar))
        
    tempo_per_bar.append(tempo_per_bar[-1])
    
    return(tempo_per_bar)


def get_harmonix_dataframe ():
    
    """
    has problems
    """
    
    PROJECT_DIR = Path(__file__).resolve().parent.parent
    HARMONIX = PROJECT_DIR / "Data/harmonix"
    
    rows = []
    for f in sorted((HARMONIX / "beats_and_downbeats").glob("*.txt")):
        tid = f.stem
        
        # each line: beat_time <tab> position_in_bar (1 = downbeat) <tab> bar_number
        data = [l.split() for l in f.read_text().splitlines() if l.strip()]
        
        #we ony have access to the times of each beat and thier position in the measure
        t = np.array([float(r[0]) for r in data])
        pos = np.array([int(float(r[1])) for r in data])
        
        
        downbeat_times = t[ np.where(pos==1) ]
        
        
        beats = beats_per_bar(pos)
        tempos = tempo_per_bar(downbeat_times, beats)
        
        rows.append(dict(
            #id
            track_id = tid,
            #identify the dataset
            dataset="harmonix",
            #path of the file if needeed (all the audios need to be in the audio folder)
            audio_path=str(HARMONIX / "audio" / f"{tid}.wav"),
            # Down beat times
            downbeat_times = downbeat_times,
            #time signature of each bar (den, num)
            beats_per_bar =beats,
            #tempos in BPM
            tempo_per_bar=tempos ,
            
            contains_meter_changes=contains_change(beats),
            contains_tempo_changes=contains_change(tempos),
            ))
        
    df_uharmonix = pd.DataFrame(rows)
    
    return(df_uharmonix)
        
d2 = get_harmonix_dataframe()