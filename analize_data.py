import argparse
import numpy as np
import pandas as pd
import os
from datasets.MIT.providers.wavedata_provider import WavedataProvider
from datasets.MIT.providers.database_provider import DatabaseProvider
from datasets.MIT.utils.name_generator import NameGenerator
from utils.helpers import flatten_list
from datasets.MIT.utils.data_structures import SliceMeta
from collections import namedtuple

DB_NAME = "mitdb"

def _load_examples(path):
    examples_dirs = (os.path.join(path, d) for d in os.listdir(path))
    examples_dirs = (d for d in examples_dirs if os.path.isdir(d))

    files = WavedataProvider()
    print("loading examples...")
    examples = [files.load(d, False)[0] for d in examples_dirs]
    examples = flatten_list(examples)

    names = NameGenerator(".csv")
    examples_meta = (names.get_metadata(e.name) for e in examples)
    examples_meta = [SliceMeta(e.record, e.rythm, int(e.start), int(e.end)) for e in examples_meta]

    df = pd.DataFrame(data=examples_meta, columns=["record", "rythm", "start", "end"])

    return df, examples

def _load_raw_data():
    database = DatabaseProvider(DB_NAME)
    print("loading raw data...")
    records = [(r.annotation, r.signal) for r in database.get_records()]

    data = []

    columns = ["record", "rythm", "beat", "start", "end"]
    Beat = namedtuple("Beat", columns)
    
    for rec, signal in records:
        rythms = (r.rstrip("\x00").lstrip("(") for r in rec.aux_note)

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

def _filter_raw_data(df):
    normal = (df.rythm == "N") & (df.beat == "N")
    left = df.beat == "L"
    rigth = df.beat == "R"
    b = df.rythm == "B"

    return df[normal | left | rigth | b]

def _filter_beats(df):
    l_beats = df[df.beat == "L"]
    r_beats = df[df.beat == "R"]
    n_beats = df[(df.rythm == "N") & ((df.beat == "N") | (df.beat == "."))]
    b_beats = df[df.rythm == "B"]

    return l_beats, r_beats, n_beats, b_beats

def _filter_examples(df):
    l_examples = df[df.rythm == "L"]
    r_examples = df[df.rythm == "R"]
    n_examples = df[df.rythm == "N"]
    b_examples = df[df.rythm == "B"]

    return l_examples, r_examples, n_examples, b_examples

def _class_length(df, name, rythm, beat, fs=360):
    if rythm:
        mask = df.rythm == rythm
    if beat:
        beat_mask = df.beat == beat
        if rythm:
            mask = mask & beat_mask
        else:
            mask = beat_mask
    
    df_filtered = df[mask]

    print("\nclass {} distribution:".format(name))
    for record, signals in df_filtered.groupby("record"):
        duration = (signals.end - signals.start).sum() / fs
        print("{}: {:.2f}s".format(record, duration))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", "-e", help="Examples root directory", type=str)
    args = parser.parse_args()

    ds, examples = _load_examples(args.examples)
    db = _load_raw_data()

    _class_length(db, "(R Right bundle branch block beat", None, "R")
    _class_length(db, "(BII 2° heart block", "BII", None)
    _class_length(db, "(SBR	Sinus bradycardia", "SBR", None)

    
    db = _filter_raw_data(db)

    print("Slice window: {}".format(os.path.dirname(args.examples)))

    #Check L/R/N distribution in dataset
    ds_groups = _filter_examples(ds)
    print("\nclass distribution in dataset:")
    for name, group in zip(["L", "R", "N", "B"], ds_groups):
        print("{}: count {},  ratio {:.3f}%".format(name, len(group), len(group) / len(ds) * 100 ))

    #Check what part of raw data from database was used in dataset
    data = {
        "record": [],
        "L": [],
        "R": [],
        "N": [],
        "B": []
    }

    for record in db.record.unique():
        db_records = db[db.record == record]
        ds_records = ds[ds.record == record]

        db_beats = _filter_beats(db_records)
        ds_rythms = _filter_examples(ds_records)

        data["record"].append(record)
        for name, db_group, ds_group in zip(["L", "R", "N", "B"], db_beats, ds_rythms):
            if (len(db_group) > 0 and len(ds_group) > 0):
                db_len = (db_group.end - db_group.start).sum()
                ds_len = (ds_group.end - ds_group.start).sum()

                ratio = ds_len / db_len * 100.0
            else:
                ratio = 0
            data[name].append(ratio)
        
    df = pd.DataFrame(data, columns=["record", "L", "R", "N", "B"])
    print("\nRatio of examples_length/total_beats_length per record:")
    print(df)

    #Total percent of class usage in dataset compared to database
    db_beats = _filter_beats(db)
    ds_rythms = _filter_examples(ds)

    print("\nClass usage in dataset compared to database:")
    for name, db_group, ds_group in zip(["L", "R", "N", "B"], db_beats, ds_rythms):
        db_len = (db_group.end - db_group.start).sum()
        ds_len = (ds_group.end - ds_group.start).sum()
        print("{}: {:.3f}".format(name, ds_len / db_len * 100))

    #Calcualte min, max, mean and var
    signal_series = pd.Series(flatten_list([e.x for e in examples]))
    print("\nSignal stats:")
    print("min: {}".format(signal_series.min()))
    print("max: {}".format(signal_series.max()))

if __name__ == "__main__":
    main()