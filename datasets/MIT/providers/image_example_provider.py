import os
from datasets.MIT.providers.base_examples_provider import BaseExamplesProvider
from datasets.MIT.providers.images_provider import ImagesProvider
from datasets.MIT.utils.data_structures import Scale
from utils.helpers import flatten_list

#based on collected dataset stats
SIG_MEAN = -0.148
SIG_STD = 1.129

class ImageExamplesProvider(BaseExamplesProvider):
    def __init__(self, params):
        super(ImageExamplesProvider, self).__init__("img", params)
        
        self.image_height = params["image_height"]
        self.sample_rate = 250

    def _build_examples(self):
        ecgs = self._get_ECGs()

        slices = [e.get_slices(self.slice_window, self.rythm_filter, self.rythm_map)  for e in ecgs]
        slices = flatten_list(slices)

        splits = self._split_slices(slices)
        images = ImagesProvider()

        for i, s in enumerate(splits):
            directory = os.path.join(self.examples_dir, str(i))
            images.save(
                    slices=s,
                    directory=directory,
                    y_range=Scale(SIG_MEAN - SIG_STD * 2, SIG_MEAN + SIG_STD * 2),
                    slice_window=self.slice_window,
                    image_height=self.image_height,
                    fs=self.sample_rate
                )
            images.augment(directory)
    
    def _load_examples(self):
        example_splits = {}
        images = ImagesProvider()

        for i in range(len(self.split_ratio)):
            directory = os.path.join(self.examples_dir, str(i))
            examples = images.load(directory, include_augmented=True)

            example_splits[i] = {
                "original": examples[0],
                "augmented": examples[1]
            }
        
        return example_splits