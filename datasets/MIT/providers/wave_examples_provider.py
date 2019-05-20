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