import os
import numpy as np
import pandas as pd
import random
from datasets.MIT.base.base_examples_provider import BaseExamplesProvider
from datasets.MIT.providers.database_provider import DatabaseProvider
from datasets.MIT.providers.wavedata_provider import WavedataProvider
from datasets.Arythmia.arythmia_ecg import ArythmiaECG
from utils.helpers import flatten_list, unzip_list
from utils.dirs import create_dirs

class WaveExamplesProvider(BaseExamplesProvider):
    def __init__(self, params):
        super(WaveExamplesProvider, self).__init__("wave", params)

    def _build_examples(self):
        ecgs = self._get_ECGs()

        slices, aug_slices = unzip_list([e.get_slices(self.slice_window, self.rythm_filter) for e in ecgs])
        slices = flatten_list(slices)
        aug_slices = flatten_list(aug_slices)

        splits = self._split_slices(slices)
        aug_splits = self._split_aug_slices(aug_slices, splits)

        wp = WavedataProvider()

        for i in range(len(splits)):
            examples, aug_examples = self._equalize_examples(splits[i], aug_splits[i])

            directory = os.path.join(self.examples_dir, str(i))
            wp.save(examples , directory)

            #TO DO: add actual augmentation
            aug_directory = os.path.join(self.examples_dir, str(i), wp.AUGMENTED_DIR)
            wp.save(aug_examples, aug_directory)

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

    def _get_ECGs(self):
        """Reads records from database and converts them to ECG objects
        Returns:
            list of ECG objects
        """
        if not self._ecgs:
            records = DatabaseProvider(self.db_name).get_records()

            self._ecgs = [ArythmiaECG(name=r.signal.record_name,
                signal=np.reshape(r.signal.p_signal, [-1]),
                labels=r.annotation.aux_note,
                beats=r.annotation.symbol,
                timecodes=r.annotation.sample) for r in records]

        return self._ecgs

    def _equalize_examples(self, examples, aug_examples):
        examples_eq = []
        aug_examples_eq = []
        
        df = pd.DataFrame(examples)  

        orig_distribution = df.rythm.value_counts().sort_values().iteritems()
        orig_distribution = list(orig_distribution)

        min_class, min_count = orig_distribution[0]
        
        examples_eq.extend([e for e in examples if e.rythm == min_class])
        aug_examples_eq.extend([e for e in aug_examples if e.rythm == min_class])

        min_count += len(aug_examples_eq) 

        for class_name, class_count in orig_distribution[1:]:
            class_examples = [e for e in examples if e.rythm == class_name]
            aug_class_examples = [e for e in aug_examples if e.rythm == class_name]

            if class_count <= min_count:
                take = class_count
                take_aug = min_count - class_count
                random.shuffle(aug_class_examples)
            else:
                take = min_count
                take_aug = 0
                random.shuffle(class_examples)

            examples_eq.extend(class_examples[:take])
            aug_examples_eq.extend(aug_class_examples[:take_aug])

        return examples_eq, aug_examples_eq   