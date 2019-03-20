import tensorflow as tf
import numpy as np
from datasets.base_dataset import BaseDataset
from datasets.MIT.database_provider import DatabaseProvider
from utils.bitmask import to_bitmask, invert_mask

TRAIN = tf.estimator.ModeKeys.TRAIN
EVAL = tf.estimator.ModeKeys.EVAL
PREDICT = tf.estimator.ModeKeys.PREDICT

class Dataset(BaseDataset):
    def __init__(self, params):
        self.params = params
        self._signal_type_to_label = {sig_type:i for i, sig_type in enumerate(params.signal_type_filter)}
        self._label_to_signal_type = [sig_type for sig_type, i in self._signal_type_to_label.items()]
        print("building dataset")
        self._dataset = self._build_dataset()
        print("building dataset complete")

    def dataset_stats(self, mode):
        print(mode)
        total_len = len(self._dataset[mode]["y"])
        for i in range(len(self.params.signal_type_filter)):
            label_num = len([label for label in self._dataset[mode]["y"] if label == i])
            print("class {} {}%".format(i, 100 * label_num / (total_len + 1e-7)))
        return

    def class_examples(self, class_name, mode):
        for example, label in zip(self._dataset[mode]["x"], self._dataset[mode]["y"]):
            if label == class_name:
                return example

    def get_input_fn(self, mode):
        def generator_fn():
            for i in range(len(self._dataset[mode]["x"])):
                yield (self._dataset[mode]["x"][i], self._dataset[mode]["y"][i])

        def input_fn():
            dataset = tf.data.Dataset.from_generator(
                generator_fn,
                (tf.float32, tf.int64),
                (tf.TensorShape((self.params.sample_len, 1)), tf.TensorShape(()))
            )

            if mode == TRAIN:
                dataset = dataset.shuffle(
                    buffer_size=len(self._dataset[mode]["x"]),
                    reshuffle_each_iteration=True
                ).repeat()
            
            batch_size = self._batch_size(mode)
            dataset = dataset.batch(batch_size)

            return dataset
            
        return input_fn

    def get_predict_data(self):
        def get_generator_fn(dataset_slice):
            def generator_fn():
                for i in range(len(dataset_slice["x"])):
                    yield dataset_slice["x"][i]

            return generator_fn

        def get_input_fn(dataset_slice):
            def input_fn():
                dataset = tf.data.Dataset.from_generator(
                    get_generator_fn(dataset_slice),
                    (tf.float32),
                    (tf.TensorShape((self.params.sample_len, 1)))
                ).batch(1)

                return dataset
            
            return input_fn

        dataset_slice = self._dataset[EVAL]
        input_fn = get_input_fn(dataset_slice)
            
        return input_fn, dataset_slice["y"]


    def _batch_size(self, mode):
        if mode == TRAIN:
            return self.params.train_batch_size
        if mode == EVAL:
            return self.params.eval_batch_size
        
    def _build_dataset(self):
        ecgs = DatabaseProvider(self.params.db_name).get_ecgs(self.params.bypass_cache)

        ecgs_samples = [{
            "name": ecg.name,
            "sample_groups": ecg.get_samples(self.params.sample_len, self.params.signal_type_filter, self.params.signal_type_map)
        } for ecg in ecgs]

        split_map = self._calculate_split_map(ecgs_samples)

        dataset = {}
       
        for set_name in [TRAIN, EVAL]:
            dataset[set_name] = {}
            dataset[set_name]["x"] = []
            dataset[set_name]["y"] = []

            for sig_type, sets in split_map.items():
                x = [ecg_samples["sample_groups"][sig_type] for ecg_samples in ecgs_samples if ecg_samples["name"] in sets[set_name]]
                x_flatten = [item for sublist in x for item in sublist]
                y = [self._signal_type_to_label[sig_type] for _ in x_flatten]
                
                dataset[set_name]["x"].extend(x_flatten)
                dataset[set_name]["y"].extend(y)

        return dataset

    def _calculate_split_map(self, ecgs_samples):
        def _groups_length(stats, mask):
            return sum(group["count"] for i, group in enumerate(stats) if mask[i] == 1)

        #dictionary like {"sig_type": ["ecg_name1", "ecg_name4", ...]}
        split_map = {}

        #dictionary like {"sig_type": {"name": "ecg_name1", "count": 17}, ...}
        label_stats = {}

        for ecg_samples in ecgs_samples:
            for sig_type, samples in ecg_samples["sample_groups"].items():
                if not sig_type in label_stats:
                    label_stats[sig_type] = []
                
                label_stats[sig_type].append({
                    "name": ecg_samples["name"],
                    "count": len(samples)
                })

        for sig_type, stats in label_stats.items():
            best_combination = []
            best_combination_error = 1.0

            groups_num = len(stats)
            total_len = sum(group["count"] for group in stats)

            if total_len > 0:
                for i in range(1, 2 ** groups_num):
                    mask = to_bitmask(i, groups_num)
                    subgroup_len = _groups_length(stats, mask)
                    curr_ratio = subgroup_len / total_len
                    curr_combination_error = abs(curr_ratio - self.params.split_ratio)

                    if curr_combination_error < best_combination_error:
                        best_combination_error = curr_combination_error
                        best_combination = mask.copy()
                
                split_map[sig_type] = {
                    TRAIN: [group["name"] for i, group in enumerate(stats) if best_combination[i] == 0],
                    EVAL: [group["name"] for i, group in enumerate(stats) if best_combination[i] == 1]
                }
            else:
                split_map[sig_type] = {
                    TRAIN: [],
                    EVAL: []
                }

        return split_map