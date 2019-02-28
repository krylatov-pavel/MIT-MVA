from database_provider import DatabaseProvider
from utils.bitmask import to_bitmask, invert_mask

class Dataset:
    def __init__(self, config):
        self.config = config
        self._dataset = None

    @property
    def train(self):
        if not self._dataset:
            self._dataset = self.__build_dataset()
        return self._dataset["train"]

    @property
    def test(self):
        if not self._dataset:
            self._dataset = self.__build_dataset()
        return self._dataset["test"]

    def __build_dataset(self):
        ecgs = DatabaseProvider(self.config.db_name).get_ecgs(self.config.bypass_cache)

        ecgs_samples = [{
            "name": ecg.name,
            "sample_groups": ecg.get_samples(self.config.sample_len, self.config.label_filter)
        } for ecg in ecgs]

        split_map = self.__calculate_split_map(ecgs_samples)

        train_x = []
        train_y = []
        test_x = []
        test_y = []

        for label, sets in split_map.items():
            train_x.extend([sample["signal"] for sample in ecgs_samples[name][label] for name in sets["train"]])
            test_x.extend([sample["signal"] for sample in ecgs_samples[name][label] for name in sets["test"]])

            train_y.extend([sample["label"] for sample in ecgs_samples[name][label] for name in sets["train"]])
            test_y.extend([sample["label"] for sample in ecgs_samples[name][label] for name in sets["test"]])

        dataset = {}
        dataset["train"] = {}
        dataset["train"]["x"] = train_x
        dataset["train"]["y"] = train_y
        dataset["test"]["x"] = test_x
        dataset["test"]["y"] = test_y

        return dataset

    def __calculate_split_map(self, ecgs_samples):
        def _groups_length(stats, mask):
            return sum(group["count"] for i, group in enumerate(stats) if mask[i] == 1)

        #dictionary like {"label": ["ecg_name1", "ecg_name4", ...]}
        split_map = {}

        #dictionary like {"label": {"name": "ecg_name1", "count": 17}, ...}
        label_stats = {}

        for ecg_samples in ecgs_samples:
            for label, samples in ecg_samples.items():
                if not label in label_stats:
                    label_stats[label] = []
                
                label_stats[label].append({
                    "name": ecg_samples["name"],
                    "count": len(samples)
                })

        for label, stats in label_stats.items():
            best_combination = []
            best_combination_error = 1.0

            groups_num = len(stats)
            total_len = sum(group["count"] for group in stats)

            for i in range(2 ** groups_num):
                mask = to_bitmask(i, groups_num)
                subgroup_len = _groups_length(stats, mask)
                curr_ratio = subgroup_len / total_len
                curr_combination_error = abs(curr_ratio - self.config.split_ratio)

                if curr_combination_error < best_combination_error:
                    best_combination_error = curr_combination_error
                    best_combination = mask.copy()
            
            split_map[label] = {
                "train": [group["name"] for i, group in enumerate(stats) if best_combination[i] == 0],
                "test": [group["name"] for i, group in enumerate(stats) if best_combination[i] == 1]
            }

        return split_map



