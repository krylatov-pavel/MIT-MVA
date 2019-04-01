import numpy as np
import pandas as pd
import os
import tensorflow as tf
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

TRAIN = tf.estimator.ModeKeys.TRAIN
EVAL = tf.estimator.ModeKeys.EVAL
PREDICT = tf.estimator.ModeKeys.PREDICT

class Dataset(BaseDataset):
    def __init__(self, params):
        self.db_name = params["db_name"]
        self.sample_len = params["sample_len"]
        self.rythm_map = params["rythm_map"]
        self.rythm_filter = params["rythm_filter"]
        self.labels_map = params["labels_map"]
        self.split_ratio = params["split_ratio"]
        self.image_height = params["image_height"]
        self.train_batch_size = params["train_batch_size"]
        self.eval_batch_size = params["eval_batch_size"]
        
        self._data = None

    def get_input_fn(self, mode):
        def get_batch_size():
            if mode == TRAIN:   
                return self.train_batch_size
            if mode == EVAL:
                return self.eval_batch_size

        def get_set_num():
            return 0 if mode == TRAIN else 0
        
        def get_subset_num():
            return 2 if mode == TRAIN else 1

        def generator_fn():
            set_num = get_set_num()
            subset_num = get_subset_num()
            
            for i in range(subset_num):
                for j in range(len(self.data[set_num][i])):
                    yield (self.data[set_num][i][j].data, self.labels_map[self.data[set_num][i][j].label])

        def input_fn():
            dataset = tf.data.Dataset.from_generator(
                generator_fn,
                (tf.float32, tf.int64),
                (tf.TensorShape((128, 181, 1)), tf.TensorShape(()))
            )

            if mode == TRAIN:
                dataset = dataset.shuffle(
                    buffer_size=len(self.data[0][0]) + len(self.data[0][1]),
                    reshuffle_each_iteration=True
                ).repeat()
            
            dataset = dataset.batch(get_batch_size())

            return dataset
            
        return input_fn

    @property
    def dataset_dir(self):
        return os.path.join("data", self.db_name, str(self.sample_len))

    @property
    def data(self):
        if not self._data:
            self._data = self._load_data()
        
        return self._data

    def _load_data(self):
        """Loads data from images
        Returns:
            3 list of Image tuples, list shape [k_folds, 2, n_samples]
        """
        data = [None] * len(self.split_ratio)
        
        if not self._dataset_exists():
            self._build_dataset()

        images = ImagesProvider()
        
        for i in range(len(self.split_ratio)):
            images_dir = os.path.join(self.dataset_dir, str(i))
            aug_images_dir = os.path.join(images_dir, images.AUGMENTED_DIR)

            data[i] = [images.load(images_dir), images.load(aug_images_dir)]

        return data
    
    def _dataset_exists(self):
        if os.path.exists(self.dataset_dir) and is_empty(self.dataset_dir):
            return False
        else:
            for i in range(len(self.split_ratio)):
                if not os.path.exists(os.path.join(self.dataset_dir, str(i))):
                    return False
        return True

    def _build_dataset(self):
        """Process database, extract labeled signals, split it into samples and convert
        to 2d grayscale images, perform data augmentation, save images to disc
        """
        print("building dataset")
        records = DatabaseProvider(self.db_name).get_records()

        if len(records) > 0:
            ecgs = [ECG(name=r.signal.record_name,
                signal=np.reshape(r.signal.p_signal, [-1]),
                labels=r.annotation.aux_note,
                timecodes=r.annotation.sample) for r in records]

            samples = [e.get_samples(self.sample_len, self.rythm_filter, self.rythm_map)  for e in ecgs]
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

            for i, fold in enumerate(fold_list):
                images_dir = os.path.join(self.dataset_dir, str(i))

                images.save(
                    samples=fold[1:128],
                    directory=images_dir,
                    y_range=Scale(SIG_MEAN - SIG_STD * 2, SIG_MEAN + SIG_STD * 2),
                    sample_len=self.sample_len,
                    image_height=self.image_height,
                    fs=records[0].signal.fs
                )

                images.augment(images_dir)

        print("building dataset complete")

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