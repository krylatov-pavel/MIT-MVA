from data_loader.database_provider import DatabaseProvider
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
            "sample_groups": ecg.get_samples(self.config.sample_len, self.config.label_filter, self.config.label_map)
        } for ecg in ecgs]

        split_map = self.__calculate_split_map(ecgs_samples)

        dataset = {}
       
        for set_name in ["train", "test"]:
            dataset[set_name] = {}
            dataset[set_name]["x"] = []
            dataset[set_name]["y"] = []

            for label, sets in split_map.items():
                x = [ecg_samples["sample_groups"][label] for ecg_samples in ecgs_samples if ecg_samples["name"] in sets[set_name]]
                x_flatten = [item for sublist in x for item in sublist]
                y = [label for _ in x_flatten]
                
                dataset[set_name]["x"].extend(x_flatten)
                dataset[set_name]["y"].extend(y)

        return dataset

    def __calculate_split_map(self, ecgs_samples):
        def _groups_length(stats, mask):
            return sum(group["count"] for i, group in enumerate(stats) if mask[i] == 1)

        #dictionary like {"label": ["ecg_name1", "ecg_name4", ...]}
        split_map = {}

        #dictionary like {"label": {"name": "ecg_name1", "count": 17}, ...}
        label_stats = {}

        for ecg_samples in ecgs_samples:
            for label, samples in ecg_samples["sample_groups"].items():
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

            for i in range(1, 2 ** groups_num):
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



