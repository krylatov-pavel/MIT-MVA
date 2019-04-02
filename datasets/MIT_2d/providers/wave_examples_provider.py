import numpy as np
import pandas as pd
import os
from datasets.MIT_2d.providers.base_examples_provider import BaseExamplesProvider
from datasets.MIT_2d.providers.wavedata_provider import WavedataProvider
from utils.helpers import flatten_list

class WaveExamplesProvider(BaseExamplesProvider):

    def _build_examples(self):
        ecgs = self._get_ECGs()

        slices = [e.get_slices(self.slice_window, self.rythm_filter, self.rythm_map)  for e in ecgs]
        slices = flatten_list(slices)

        splits = self._split_slices(slices)

        for i, s in enumerate(splits):
            directory = os.path.join(self.examples_dir, str(i))
            WavedataProvider.save(s, directory)

    def _load_examples(self):
        example_splits = {}

        for i in range(len(self.split_ratio)):
            directory = os.path.join(self.examples_dir, str(i))
            examples = WavedataProvider.load(directory, include_augmented=True)

            example_splits[i] = {
                "regular": examples[0],
                "augmented": examples[1]
            }
        
        return example_splits