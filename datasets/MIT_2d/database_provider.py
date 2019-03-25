import wfdb
import os
import pickle
from utils.dirs import create_dirs, is_empty, clear_dir

class DatabaseProvider(object):
    def __init__(self, db_name):
        self.db_name = db_name

    def ged_records(self, bypass_cache=False):
        db_dir = os.path.join("data", "database", self.db_name, "records")

        if not os.path.exists:
            create_dirs([db_dir])

        if is_empty(db_dir) or bypass_cache:
            clear_dir(db_dir)
            self._save_records(db_dir)
        
        return self._load_records(db_dir)

    def _save_records(self, db_dir):
        record_names = wfdb.io.get_record_list(self.db_name, records="all")

        record_data = [self.__fetch_record(name) for name in record_names]
        
        for name, data in zip(record_names, record_data):
            if data != None:
                record_dir = os.path.join(db_dir, name)
                create_dirs([record_dir])

                #serialize signal object to /{database_name}/{record_name}/signal.pkl
                signal_fpath = os.path.join(record_dir, "signal.pkl")
                with open(signal_fpath, "wb") as f:
                    pickle.dump(data[0], f, pickle.DEFAULT_PROTOCOL)

                #serialize signal annotation object to /{database_name}/{record_name}/annotation.pkl
                ann_fpath = os.path.join(record_dir, "annotation.pkl")
                with open(ann_fpath, "wb") as f:
                    pickle.dump(data[1], f, pickle.DEFAULT_PROTOCOL)

    #
    def _load_records(self, db_dir):
        """ Loads list of records data from files in db_dir directory
        Returns dictionary with record name as Key and record data dictionary as Value
        {
            "418": {
                "signal" : wfdb's rdrecord object,
                "annotation": wfdb's rdann object 
            }
        } 
        """
        
        raise NotImplementedError()

    def __fetch_record(self, record_name):
        try:
            rec_data = wfdb.rdrecord(record_name, pb_dir=self.db_name, channels=[0], physical=True)
            rec_annotation = wfdb.io.rdann(record_name, extension = "atr", pb_dir=self._db_name)
            
            print("downloaded record {} data".format(record_name))
            return rec_data, rec_annotation
        except:
            print("error downloading record {} data".format(record_name))
            return None

        