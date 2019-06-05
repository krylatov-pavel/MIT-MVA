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
PLOT_PATH = "data\\stats"
FS = 360
BEATS = [
    #(["N", "·"], "Normal beat", "N"),
    (["L"], "Left bundle branch block beat", "L"),
    (["R"], "Right bundle branch block beat", "R"),
    (["A"], "Atrial premature beat", "A"),
    (["a"], "Aberrated atrial premature beat", "a"),
    (["J"], "Nodal (junctional) premature beat", "J"),
    (["S"], "Supraventricular premature beat", "S"),
    (["V"], "Premature ventricular contraction", "V"),
    (["F"], "Fusion of ventricular and normal beat", "F"),
    (["!"], "Ventricular flutter wave", "!"),
    (["e"], "Atrial escape beat", "e"),
    (["j"], "Nodal (junctional) escape beat", "j"),
    (["E"], "Ventricular escape beat", "E"),
    (["/"], "Paced beat", "/"),
    (["f"], "Fusion of paced and normal beat", "f"),
    (["x"], "Non-conducted P-wave (blocked APB)", "x")

]
RYTHMS = [
    ("(AB", "Atrial bigeminy", "AB"),
    ("(AFIB", "Atrial fibrillation", "AFIB"),
    ("(AFL", "Atrial flutter", "AFL"),
    ("(B", "Ventricular bigeminy", "B"),
    ("(BII", "2° heart block", "BII"),
    ("(IVR", "Idioventricular rhythm", "IVR"),
    #("(N", "Normal sinus rhythm", "N"),
    ("(NOD", "Nodal (A-V junctional) rhythm", "NOD"),
    ("(P", "Paced rhythm", "P"),
    ("(PREX", "Pre-excitation (WPW)", "PREX"),
    ("(SBR", "Sinus bradycardia", "SBR"),
    ("(SVTA", "Supraventricular tachyarrhythmia", "SVTA"),
    ("(T", "Ventricular trigeminy", "T"),
    ("(VFL", "Ventricular flutter", "VFL"),
    ("(VT", "Ventricular tachycardia", "VT")
]

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

def _plot_all(all_stats, name, config):
    create_dirs([os.path.join(PLOT_PATH, DB_NAME)])

    all_stats.sort(key=lambda x: x[1], reverse=True)

    classes = [s[0] for s in all_stats]
    durations = [s[1] for s in all_stats]

    x_pos = np.arange(len(classes))
    y_pos =  np.arange(max(durations) // 300 + 1) * 300.0
    
    fig = plt.figure(figsize=(20,10))
    plt.bar(x_pos, durations)

    plt.xticks(x_pos, classes)
    plt.yticks(y_pos, [_format_sec(sec) for sec in y_pos])

    plt.ylabel("Duration")
    plt.xlabel("Classes")

    plt.title("All {}".format(name))
    plt.grid(axis="y")

    legend = []
    for c in classes:
        description = [d for d in config if d[2] == c][0]
        legend.append("{} - {}".format(description[2], description[1]))
    legend = "\n".join(legend)
    props = dict(boxstyle='round', facecolor='wheat')
    x = int(0.70 * len(classes))
    y = int(0.5 * durations[0])
    plt.text(x, y, legend, fontsize=14, bbox=props)

    fname = "all_{}.png".format(name)
    fpath = os.path.join(PLOT_PATH, DB_NAME, fname)
    plt.savefig(fpath)   

    plt.close(fig)

def _plot_class(df, dir, name, records):
    create_dirs([os.path.join(PLOT_PATH, DB_NAME, dir)])

    durations = []
    
    for rec in records:
        beats_in_record = df[df.record == rec]
        duration = (beats_in_record.end - beats_in_record.start).sum() if len(beats_in_record) > 0 else 0
        durations.append(duration / FS)
    total_duration = _format_sec(sum(durations))

    x_pos = np.arange(len(records))
    y_pos = np.arange(16) * 120.0
    
    fig = plt.figure(figsize=(20,10))
    plt.bar(x_pos, durations)

    plt.xticks(x_pos, records, rotation=90)
    plt.yticks(y_pos, [_format_sec(sec) for sec in y_pos])

    plt.ylabel("Duration(sec)")
    plt.xlabel("Records")

    plt.title("{} (total: {})".format(name, total_duration))
    plt.grid(axis="y")

    fname = "{}.png".format(name)
    fpath = os.path.join(PLOT_PATH, DB_NAME, dir, fname)
    plt.savefig(fpath)
    plt.close(fig)

def _plot_combinations(df, rhythm=None, beat=None):
    if rhythm:
        series = df.beat
        dir = "rhythms"
        title = "Combination of {} rhythm with beats".format(rhythm)
    else:
        series = df.rythm
        dir = "beats"
        title = "Combination of {} beat with rhythms".format(beat)

    stats = list(series.value_counts(normalize=True).iteritems())
    labels = [s[0] for s in stats]
    values = [s[1] for s in stats]

    x_pos = np.arange(len(labels))
    y_pos = np.arange(11) * 0.1
    
    fig = plt.figure(figsize=(15,7))

    plt.bar(x_pos, values, width=0.9)

    plt.xlim((-1, 15))
    plt.ylim((0, 1.0))

    plt.xticks(x_pos, labels)
    plt.yticks(y_pos, [str(int(percent * 100)) for percent in y_pos])

    plt.ylabel("Percent(%)")
    plt.xlabel(dir)

    plt.title(title)
    plt.grid(axis="y")

    fname = "{}.png".format(rhythm or beat)
    path = os.path.join(PLOT_PATH, DB_NAME, "combinations", dir)
    create_dirs([path])
    fpath = os.path.join(path, fname)
    plt.savefig(fpath)
    plt.close(fig)        

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
    
    all_beats = []
    all_rythms = []

    for b in BEATS:
        beats = _filter(df, beats=b[0])
        _plot_class(beats, dir="beats", name=b[1], records=records)
        _plot_combinations(beats, beat=b[2])
        all_beats.append((b[2], (beats.end - beats.start).sum() / FS))
    
    for r in RYTHMS:
        beats = _filter(df, rythm=r[0])
        _plot_class(beats, dir="rythms", name=r[1], records=records)
        _plot_combinations(beats, rhythm=r[2])
        all_rythms.append((r[2], (beats.end - beats.start).sum() / FS))

    _plot_all(all_beats, "beats", BEATS)
    _plot_all(all_rythms, "rythms", RYTHMS)

if __name__ == "__main__":
    main()