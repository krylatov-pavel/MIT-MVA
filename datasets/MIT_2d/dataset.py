import numpy as np
import pandas as pd
import os
from datasets.base_dataset import BaseDataset
from datasets.MIT_2d.database_provider import DatabaseProvider
from datasets.MIT_2d.ecg import ECG
from datasets.MIT_2d.images_provider import ImagesProvider
from datasets.MIT_2d.combinator import Combinator
from datasets.MIT_2d.data_structures import Scale
from utils.helpers import flatten_list
from utils.dirs import is_empty, clear_dir, create_dirs
from utils.pd_utils import list_max, list_min

#based on collected dataset stats
SIG_MEAN = -0.148
SIG_STD = 1.129

class Dataset(BaseDataset):
    def __init__(self, params):
        self.db_name = params["db_name"]
        self.sample_len = params["sample_len"]
        self.labels_map = params["labels_map"]
        self.labels_filter = params["labels_filter"]
        self.split_ratio = params["split_ratio"]
        self.image_height = params["image_height"]

        if not self._dataset_exists():
            self._build_dataset()

    def get_input_fn(self, mode):
        raise NotImplementedError()

    @property
    def dataset_dir(self):
        return os.path.join("data", self.db_name, str(self.sample_len))

    def _dataset_exists(self):
        if os.path.exists(self.dataset_dir) and is_empty(self.dataset_dir):
            return False
        else:
            for i in range(len(self.split_ratio)):
                if not os.path.exists(os.path.join(self.dataset_dir, str(i))):
                    return False
        return True

    def _build_dataset(self):
        records = DatabaseProvider(self.db_name).get_records()

        if len(records) > 0:
            ecgs = [ECG(name=r.signal.record_name,
                signal=np.reshape(r.signal.p_signal, [-1]),
                labels=r.annotation.aux_note,
                timecodes=r.annotation.sample) for r in records]

            samples = [e.get_samples(self.sample_len, self.labels_filter, self.labels_map)  for e in ecgs]
            samples = flatten_list(samples)

            samples_df = pd.DataFrame(samples, columns=["record", "rythm", "start", "end", "signal"])

            split_map = self._build_split_map(samples_df, self.split_ratio)

            fold_list = []
            for k in range(len(self.split_ratio)):
                fold = []
                for rythm, group in samples_df.groupby("rythm"):
                    include_records = group.record.isin(split_map[rythm][k])
                    rythm_samples = [s for s in group[include_records].itertuples()]
                    
                    fold.extend(rythm_samples)
                
                fold_list.append(fold)
            
            images = ImagesProvider()

            images.convert_to_images(
                sample_groups=[fold_list[1]],
                dataset_dir=self.dataset_dir,
                y_range=Scale(SIG_MEAN - SIG_STD * 2, SIG_MEAN + SIG_STD * 2),
                sample_len=self.sample_len,
                image_height=self.image_height,
                fs=records[0].signal.fs
            )

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

        for rythm, rythm_group in df.groupby("rythm"):
            samples = [(record, len(record_group)) for record, record_group in rythm_group.groupby("record")]
            samples_splitted = combinator.split(samples, split_ratio)
            split_map[rythm] = [[s[0] for s in subgroup] for subgroup in samples_splitted]
        
        return split_map

    def signal_stats(self):
        records = DatabaseProvider(self.db_name).get_records()
        signal = flatten_list([r.signal.p_signal for r in records])
        
        sr = pd.Series(signal)

        print("mean: ", sr.mean()) 
        print("stddev: ", sr.std())
        print("median: ", sr.median())
        print("min:", sr.min())
        print("max:", sr.max())