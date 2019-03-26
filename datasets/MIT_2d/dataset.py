import numpy as np
import pandas as pd
from datasets.base_dataset import BaseDataset
from datasets.MIT_2d.database_provider import DatabaseProvider
from datasets.MIT_2d.ecg import ECG
from utils.helpers import flatten_list
from datasets.MIT_2d.combinator import Combinator

class Dataset(BaseDataset):
    def __init__(self, params):
        self.db_name = params["db_name"]
        self.sample_len = params["sample_len"]
        self.labels_map = params["labels_map"]
        self.labels_filter = params["labels_filter"]
        self.split_ratio = params["split_ratio"]

    def test(self):
        records = DatabaseProvider(self.db_name).get_records()
        ecgs = [ECG(name=r.signal.record_name,
            signal=np.reshape(r.signal.p_signal, [-1]),
            labels=r.annotation.aux_note,
            timecodes=r.annotation.sample) for r in records]

        samples = [e.get_samples(self.sample_len, self.labels_filter, self.labels_map)  for e in ecgs]
        samples = flatten_list(samples)

        samples_df = pd.DataFrame(samples, columns=["Record", "Rythm", "Start", "End"])
        split_map = self._build_split_map(samples_df, self.split_ratio)

        fold_list = []
        for k in range(len(self.split_ratio)):
            fold = []
            for rythm, group in samples_df.groupby("Rythm"):
                include_records = group.Record.isin(split_map[rythm][k])
                rythm_samples = [s for s in group[include_records].itertuples()]
                
                fold.extend(rythm_samples)
            
            fold_list.append(fold)

        for fold in fold_list:
            print("samples in fold: {}".format(len(fold)))

        print("Sanity check:\n", fold_list[0][0])
        
    def _build_split_map(self, df, split_ratio):
        """
        Returns: dictionary with rythm type keys, and k-length 2d list values, e.g:
        {
            "(N": [["418", "419"], ["500"], ...],
            ...
        }  
        """
        split_map = {}
        combinator = Combinator()

        for rythm, rythm_group in df.groupby("Rythm"):
            samples = [(record, len(record_group)) for record, record_group in rythm_group.groupby("Record")]
            samples_splitted = combinator.split(samples, split_ratio)
            split_map[rythm] = [[s[0] for s in subgroup] for subgroup in samples_splitted]
        
        return split_map

    def get_input_fn(self, mode):
        raise NotImplementedError()