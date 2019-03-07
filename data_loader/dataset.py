import tensorflow as tf
import numpy as np
from data_loader.database_provider import DatabaseProvider
from utils.bitmask import to_bitmask, invert_mask

class Dataset:
    def __init__(self, config):
        self.config = config
        self._signal_type_to_label = {sig_type:i for i, sig_type in enumerate(config.signal_type_filter)}
        self._label_to_signal_type = [sig_type for sig_type, i in self._signal_type_to_label.items()]
        self._dataset = None

    def get_input_fn(self, mode):
        def generator_fn():
            for i in range(len(self._dataset[mode]["x"])):
                yield (self._dataset[mode]["x"][i], self._dataset[mode]["y"][i])

        def input_fn():
            if not self._dataset:
                print("building dataset")
                self._dataset = self._build_dataset()
                print("building dataset complete")
            
            #features = self._dataset[mode]["x"]
            #labels = self._dataset[mode]["y"]
            #features = np.random.random((1000, 750, 1))
            #labels = np.ones((1000))

            batch_size = self._batch_size(mode)
            
            #dataset = tf.data.Dataset.from_tensor_slices((features, labels))
            dataset = tf.data.Dataset.from_generator(
                generator_fn,
                (tf.float32, tf.int64),
                (tf.TensorShape((750, 1)), tf.TensorShape(()))
            )

            if mode == tf.estimator.ModeKeys.TRAIN:
                dataset = dataset.shuffle(len(self._dataset[mode]["x"]))
                dataset = dataset.repeat()
            
            dataset = dataset.batch(batch_size)

            return dataset
            
        return input_fn

    def _batch_size(self, mode):
        if mode == tf.estimator.ModeKeys.TRAIN:
            return self.config.train_batch_size
        elif mode == tf.estimator.ModeKeys.EVAL:
            return self.config.eval_batch_size
        
    def _build_dataset(self):
        ecgs = DatabaseProvider(self.config.db_name).get_ecgs(self.config.bypass_cache)

        ecgs_samples = [{
            "name": ecg.name,
            "sample_groups": ecg.get_samples(self.config.sample_len, self.config.signal_type_filter, self.config.signal_type_map)
        } for ecg in ecgs]

        split_map = self._calculate_split_map(ecgs_samples)

        dataset = {}
       
        for set_name in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
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

            for i in range(1, 2 ** groups_num):
                mask = to_bitmask(i, groups_num)
                subgroup_len = _groups_length(stats, mask)
                curr_ratio = subgroup_len / total_len
                curr_combination_error = abs(curr_ratio - self.config.split_ratio)

                if curr_combination_error < best_combination_error:
                    best_combination_error = curr_combination_error
                    best_combination = mask.copy()
            
            split_map[sig_type] = {
                tf.estimator.ModeKeys.TRAIN: [group["name"] for i, group in enumerate(stats) if best_combination[i] == 0],
                tf.estimator.ModeKeys.EVAL: [group["name"] for i, group in enumerate(stats) if best_combination[i] == 1]
            }

        return split_map