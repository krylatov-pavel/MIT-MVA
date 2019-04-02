import os
import numpy as np
import pandas as pd
from datasets.MIT_2d.database_provider import DatabaseProvider
from datasets.MIT_2d.ecg import ECG
from datasets.MIT_2d.utils import Combinator
from utils.dirs import is_empty

class BaseExamplesProvider(object):
    def __init__(self, name, params):
        self.name = name

        self.db_name = params["db_name"]
        self.slice_window = params["slice_window"]
        self.rythm_map = params["rythm_map"]
        self.rythm_filter = params["rythm_filter"]
        self.labels_map = params["labels_map"]
        self.split_ratio = params["split_ratio"]

        self._ecgs = None
        self._examples = None

    #abstract members
    def _build_examples(self):
        """process records, creates labeled examples and saves them to disk
        returns: None
        """
        raise NotImplementedError()

    def _load_examples(self):
        """load examples from disk
        returns: dictionary of Example namedtupe 
        {
            {split_number}: {
                "regular": [Example, ...]
                "augmented": [Example, ...]
            },
            ...
        }
        """
        raise NotImplementedError()

    #base members
    def get(self, splits, augmented):
        """Gets examples set from splits numbers list
        Args:
            splits: list of int, numbers of splits.
            In default scenario [0] for TRAIN set, [1] for EVAL
            In k-split validation scenario, for example, for 5-split validation it could be [0, 1, 2, 3] for TRAIN
            and [4] for EVAL
            augmented: bool. True for TRAIN, False for EVAL
        Returns:
            list of Example named tuple (x, y)
        """
        if not self._examples:
            if not self.examples_exists:
                self._build_examples()
            self._examples = self._load_examples()
        
        #TO DO: read examples and return set

    @property
    def examples_dir(self):
        return os.path.join("data", self.db_name, self.name, str(self.slice_window))

    @property
    def examples_exists(self):
        if os.path.exists(self.examples_dir) and is_empty(self.examples_dir):
            return False
        else:
            for i in range(len(self.split_ratio)):
                if not os.path.exists(os.path.join(self.examples_dir, str(i))):
                    return False
        return True

    def _get_ECGs(self):
        """Reads records from database and converts them to ECG objects
        Returns:
            list of ECG objects
        """
        if not self._ecgs:
            records = DatabaseProvider(self.db_name).get_records()

            self._ecgs = [ECG(name=r.signal.record_name,
                signal=np.reshape(r.signal.p_signal, [-1]),
                labels=r.annotation.aux_note,
                timecodes=r.annotation.sample) for r in records]

        return self._ecgs

    def _split_slices(self, slices):
        """Split slices with according to slpit_ratio distribution.
        Args:
            slices: list of Slice namedtuple
        In default case, [0.8, 0.2], 80% for TRAIN set and 20% for EVAL
        In case k-fold validation, [0.2, 0.2, 0.2, 0.2, 0.2] each fold has 20% of examples
        Returns:
            list of shape [slice_num, n_slices]
        """
        slices_df = pd.DataFrame(slices, columns=["record", "rythm", "start", "end", "signal"])

        split_map = self.__build_split_map(slices_df)

        splits_list = []
        for k in range(len(self.split_ratio)):
            split = []
            for rythm, group in slices_df.groupby("rythm"):
                include_records = group.record.isin(split_map[rythm][k])
                rythm_slices = [s for s in group[include_records].itertuples()]
                
                split.extend(rythm_slices)
            
            splits_list.append(split)
        
        return splits_list

    def __build_split_map(self, df):
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
            slices = [(record, len(record_group)) for record, record_group in rythm_group.groupby("record")]
            slices_splitted = combinator.split(slices, self.split_ratio)
            split_map[rythm] = [[s[0] for s in subgroup] for subgroup in slices_splitted]
        
        return split_map