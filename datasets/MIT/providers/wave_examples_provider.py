import os
from datasets.MIT.base.base_examples_provider import BaseExamplesProvider
from datasets.MIT.providers.wavedata_provider import WavedataProvider
from utils.helpers import flatten_list

class WaveExamplesProvider(BaseExamplesProvider):
    def __init__(self, params):
        super(WaveExamplesProvider, self).__init__("wave", params)

    def _build_examples(self):
        ecgs = self._get_ECGs()

        slices = [e.get_slices(self.slice_window, self.rythm_filter, self.rythm_map)  for e in ecgs]
        slices = flatten_list(slices)

        splits = self._split_slices(slices)
        wp = WavedataProvider()

        for i, s in enumerate(splits):
            directory = os.path.join(self.examples_dir, str(i))
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