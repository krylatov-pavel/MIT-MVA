import numpy as np
import pandas as pd
import datetime
import os
import datetime
import matplotlib.pyplot as plt
from datasets.MIT.providers.database_provider import DatabaseProvider
from collections import namedtuple
from utils.dirs import create_dirs

DB_NAME = "mitdb"
FS = 360
BEATS = [
    (["L"], "Left bundle branch block beat"),
    (["R"], "Right bundle branch block beat"),
    (["A"], "Atrial premature beat")
]
RYTHMS = [
    ("(AB", "Atrial bigeminy")
]
PLOT_PATH = "data\\stats"

def _filter(df, rythm=None, beats=[]):
    mask = None

    filter_by_rythm = rythm
    filter_by_beat = beats and len(beats) > 0
    if filter_by_rythm:
        rythm_mask = df.rythm == rythm
        mask = rythm_mask
    if filter_by_beat:
        beats_mask = df.beat.isin(beats)
        if filter_by_rythm:
            mask = mask & beats_mask
        else:
            mask = beats_mask

    return df[mask]

def _format_sec(sec):
    return str(datetime.timedelta(seconds=sec)).split(".")[0]

def _histogram(df, name, records):
    durations = []
    
    for rec in records:
        beats_in_record = df[df.record == rec]
        duration = (beats_in_record.end - beats_in_record.start).sum() if len(beats_in_record) > 0 else 0
        durations.append(duration / FS)
    total_duration = _format_sec(sum(durations))

    x_pos = np.arange(len(records))
    y_pos = np.arange(16) * 120.0
    
    plt.figure(figsize=(20,10))
    plt.bar(x_pos, durations)

    plt.xticks(x_pos, records, rotation=90)
    plt.yticks(y_pos, [_format_sec(sec) for sec in y_pos])

    plt.ylabel("Duration(sec)")
    plt.xlabel("Records")

    plt.title("{} (total: {})".format(name, total_duration))
    plt.grid(axis="y")

    fname = "{}.png".format(name)
    fpath = os.path.join(PLOT_PATH, DB_NAME, fname)
    plt.savefig(fpath)

def _load_raw_data():
    database = DatabaseProvider(DB_NAME)
    print("loading raw data...")
    records = [(r.annotation, r.signal) for r in database.get_records()]

    data = []

    columns = ["record", "rythm", "beat", "start", "end"]
    Beat = namedtuple("Beat", columns)
    
    for rec, signal in records:
        rythms = (r.rstrip("\x00") for r in rec.aux_note)

        curr_rythm = None
        prev_start = 0
        for rythm, beat, start, end in zip(rythms, rec.symbol, rec.sample, np.append(rec.sample[1:], len(signal.p_signal))):
            if rythm:
                curr_rythm = rythm

            s = prev_start + ((start - prev_start) // 2)
            e = start + ((end - start) // 2)
            
            beat = Beat(rec.record_name, curr_rythm, beat, s, e)
            data.append(beat)

            prev_start = start

    df = pd.DataFrame(data, columns=columns)

    return df

def main():
    df = _load_raw_data()
    records = df.record.unique()
    
    create_dirs([os.path.join(PLOT_PATH, DB_NAME)])

    for b in BEATS:
        beats = _filter(df, beats=b[0])
        _histogram(beats, name=b[1], records=records)
    
    for r in RYTHMS:
        beats = _filter(df, rythm=r[0])
        _histogram(beats, name=r[1], records=records)

if __name__ == "__main__":
    main()