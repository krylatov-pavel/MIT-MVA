import os
import wfdb
import pickle
from ecg import ECG
from utils.dirs import create_dirs, is_empty, clear_dir 

class DatabaseProvider:
    def __init__(self, db_name):
        self._db_name = db_name

    def get_ecgs(self, bypass_cache=False):
        db_path = os.path.join("data", "database", self._db_name, )
        create_dirs([db_path])

        if is_empty(db_path) or bypass_cache:
            record_list = wfdb.io.get_record_list(self._db_name, records="all")
            ecg_list = [self.__fetch_ecg(record_name) for record_name in record_list]

            self.__save_ecgs(ecg_list, db_path)
            #do not return ecg_list here just to make sure script always loads them from files 

        return self.__load_ecgs(db_path)

    def __fetch_ecg(self, record_name):
        rec_data = wfdb.rdrecord(record_name, pb_dir=self._db_name, physical=True)
        rec_annotation = wfdb.io.rdann(record_name, extension = "atr", pb_dir=self._db_name)
        print("downloaded {} record data".format(record_name))

        if len(rec_annotation.aux_note) != len(rec_annotation.sample):
            raise IndexError("Sample length and aux_note length not match. Reocrd name: {}".format(record_name))

        annotations = [{
            "label": ann[0].rstrip["\x00"],
            "start": ann[1],
            "end": ann[2]
        } for ann in zip(rec_annotation.aux_note, rec_annotation.sample, rec_annotation.sample[1:].append(rec_data.sig_len))]

        return ECG(record_name, rec_annotation.sample, annotations)

    def __save_ecgs(self, ecg_list, db_path):
        fpath = os.path.join(db_path, "records.pkl")
        with open(fpath, "wb") as f:
            pickle.dump(ecg_list, f, pickle.DEFAULT_PROTOCOL)

    def __load_ecgs(self, db_path):
        fpath = os.path.join(db_path, "records.pkl")
        with open(fpath, "rb") as f:
            return pickle.load(f)    