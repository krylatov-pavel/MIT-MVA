import csv
import os
import numpy as np
from datasets.MIT.utils.data_structures import Example
from datasets.MIT.utils.name_generator import NameGenerator
from utils.dirs import is_empty, clear_dir, create_dirs

class WavedataProvider(object):
    def __init__(self):
        self.AUGMENTED_DIR = "augmented"

    def save(self, slices, directory):
        """Save slices to disk in a proper file format
        Args:
            slices: list of Slice namedpuples
            directory: directory to save slices
        """
        if os.path.exists(directory) and not is_empty(directory):
            clear_dir(directory)
        else:
            create_dirs([directory])
        
        generator = NameGenerator(".csv")

        for s in slices:
            fname = generator.generate_name(
                index=s.Index,
                rythm=s.rythm,
                record=s.record,
                start=s.start,
                end=s.end
            )
            fpath = os.path.join(directory, fname)

            with open(fpath, "w", newline='') as f:
                wr = csv.writer(f)
                wr.writerows(np.expand_dims(s.signal, axis=1))

    def load(self, directory, include_augmented=False):
        """Loads examples from disk
        Args:
            directory: target directory
            include_augmented: if True, return augmented images as secod element of returned list
        Returns:
            ([regular_examples], <[augmented_examples]>), elemets are Example naedpuples
        """
        examples = self._load_dir(directory)
        examples_aug = []
        if include_augmented:
            examples_aug = self._load_dir(os.path.join(directory, self.AUGMENTED_DIR))

        return (examples, examples_aug)

    def _load_dir(self, directory):
        if os.path.exists(directory):
            fnames = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
            signals = [None] * len(fnames)
            labels = [None] * len(fnames)

            generator = NameGenerator(".csv")

            for i, fname in enumerate(fnames):
                try:
                    fpath = os.path.join(directory, fname)
                    with open(fpath, "r", newline='') as f:
                        reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
                        signal = list(reader)

                    label = generator.get_rythm(fname)
                    if label:
                        labels[i] = label
                        signals[i] = signal
                    else:
                        print("Skipped file {}, can't parse name".format(fpath))
                except Exception as e:
                    print("Skipped file {}, can't read image".format(fpath))
                    if hasattr(e, 'message'):
                        print(e.message)
                    else:
                        print(e)
        else:
            print("directory not exists: ", directory)
            return []

        filtered = [Example(s, lbl, f) for s, lbl, f in zip(signals, labels, fnames) if not (s is None) and bool(lbl)]

        return filtered