import numpy as np
import pandas as pd
from datasets.base_dataset import BaseDataset
from datasets.MIT_2d.database_provider import DatabaseProvider
from datasets.MIT_2d.ecg import ECG
from utils.helpers import flatten_list

class Dataset(BaseDataset):
    def __init__(self, params):
        self.db_name = params["db_name"]
        self.sample_len = params["sample_len"]
        self.labels_map = params["labels_map"]
        self.labels_filter = params["labels_filter"]

    def test(self):
        records = DatabaseProvider(self.db_name).get_records()
        ecgs = [ECG(name=r.signal.record_name,
            signal=np.reshape(r.signal.p_signal, [-1]),
            labels=r.annotation.aux_note,
            timecodes=r.annotation.sample) for r in records]

        samples = [e.get_samples(self.sample_len, self.labels_filter, self.labels_map)  for e in ecgs]
        samples = flatten_list(samples)

        samples_df = pd.DataFrame(samples, columns=["Record", "Rythm", "Start", "End"])

        print(samples_df.head())

    def get_input_fn(self, mode):
        raise NotImplementedError()