import os
import numpy as np
from datasets.MIT.base.base_examples_provider import BaseExamplesProvider
from datasets.MIT.providers.database_provider import DatabaseProvider
from datasets.MIT.providers.wavedata_provider import WavedataProvider
from datasets.Arythmia.arythmia_ecg import ArythmiaECG
from utils.helpers import flatten_list

class WaveExamplesProvider(BaseExamplesProvider):
    def __init__(self, params):
        super(WaveExamplesProvider, self).__init__("wave", params)

    def _build_examples(self):
        ecgs = self._get_ECGs()

        slices = [e.get_slices(self.slice_window, self.rythm_filter)  for e in ecgs]
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