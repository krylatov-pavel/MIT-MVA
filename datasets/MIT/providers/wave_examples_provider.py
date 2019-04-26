import os
import numpy as np
import pandas as pd
import random
from datasets.MIT.base.base_examples_provider import BaseExamplesProvider
from datasets.MIT.providers.wavedata_provider import WavedataProvider
from utils.helpers import flatten_list

class WaveExamplesProvider(BaseExamplesProvider):
    def __init__(self, params):
        super(WaveExamplesProvider, self).__init__("wave", params)
        self.test_set_size = params["test_set_size"]

    def _build_examples(self):
        ecgs = self._get_ECGs()

        slices = [e.get_slices(self.slice_window, self.rythm_filter, self.rythm_map)  for e in ecgs]
        slices = flatten_list(slices)

        splits = self._split_slices(slices)
        wp = WavedataProvider()

        for i, s in enumerate(splits):
            random.shuffle(s)

            test_examples_num = int(self.test_set_size * len(s))
            test_directory = os.path.join(self.examples_dir, wp.TEST_DIR)
            wp.save(s[:test_examples_num], test_directory)

            directory = os.path.join(self.examples_dir, str(i))
            wp.save(s[test_examples_num:], directory)

        aug_slices = [e.get_slices(self.slice_window, self.rythm_filter, self.rythm_map, resample=True)  for e in ecgs]
        aug_slices = flatten_list(aug_slices)

        aug_splits = self._split_aug_slices(aug_slices, splits)

        for i, s in enumerate(aug_splits):
            directory = os.path.join(self.examples_dir, str(i), wp.AUGMENTED_DIR)
            wp.save(s, directory)

    def _load_examples(self):
        example_splits = {}
        wp = WavedataProvider()

        for i in range(len(self.split_ratio)):
            directory = os.path.join(self.examples_dir, str(i))
            examples = wp.load(directory, include_augmented=True)

            example_splits[i] = {
                "original": examples[0],
                "augmented": examples[1]
            }
        
        return example_splits

    def _split_aug_slices(self, slices, original_splits):
        """Split augmented slices according distribution of original slices.
        Args:
            slices: list of Slice namedtuple
            original_splits: k-length list of splits, each split contains slices
        Returns:
            list of shape [k, n_slices]
        """
        aug_splits = [None] * len(original_splits)

        df_columns = ["record", "rythm", "start", "end", "signal"]
        aug_slices_df = pd.DataFrame(slices, columns=df_columns)

        for i, s in enumerate(original_splits):
            slices_df = pd.DataFrame(s, columns=["index"] + df_columns)

            aug_split_slices = []
            aug_splits[i] = aug_split_slices

            for rythm, group in slices_df.groupby("rythm"):
                records = group["record"].unique()

                include = aug_slices_df[(aug_slices_df["rythm"] == rythm) & (aug_slices_df["record"].isin(records))]

                aug_split_slices.extend(include.itertuples())


            #equalize distribution of classes
            """
            orig_distribution = slices_df["rythm"].value_counts().sort_values().iteritems()
            orig_distribution = list(orig_distribution)

            min_class, min_count = orig_distribution[0]

            min_class_slices = [s for s in aug_split_slices if s.rythm == min_class]
            aug_splits[i].extend(min_class_slices)

            min_class_slices_num = min_count + len(min_class_slices) #original + augmented

            for j in range(1, len(orig_distribution)):
                class_name, class_count = orig_distribution[j]

                take = min_class_slices_num - class_count
                if take > 0:
                    class_slices = [s for s in aug_split_slices if s.rythm == class_name]
                    random.shuffle(class_slices)

                    take = min_class_slices_num - class_count

                    aug_splits[i].extend(class_slices[:take])
            """

        return aug_splits