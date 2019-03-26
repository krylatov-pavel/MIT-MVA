import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from utils.dirs import create_dirs, clear_dir, is_empty

class ImagesProvider(object):
    def convert_to_images(self, sample_groups, dataset_dir):
        """Converts sample list to images, saves them to disc
        Args:
            sample_groups: 2d list, length k, where k is number of groups,
            elements are namedtuples, (Index, rythm, start, end, signal), e.g:
            [[(rythm="(N", start=10, end=760, signal=[0.222, 0.225, ...]), (...)], ...]
            dataset_dir: directory to save/load files
        Returns:
            None
        """

        if os.path.exists(dataset_dir) and not is_empty(dataset_dir):
            clear_dir(dataset_dir)

        group_dirs = [os.path.join(dataset_dir, str(i)) for i in range(len(sample_groups))]
        create_dirs(group_dirs)

        for i, group in enumerate(sample_groups):
            for sample in group:
                self._save_image(group_dirs[i],
                    index=sample.Index,
                    rythm=sample.rythm,
                    signal=sample.signal
                )


    def _save_image(self, dataset_dir, index, rythm, signal):
        """Plots signal and saves it as an image to disk
        Args:
            dataset_dir: directory path
            index: number of image to avoid name collision, e.g 301
            rythm: rythm type, e.g "(N"
            signal: list of sample_len length, e.g [0.222, 0.225, ...]
        Returns: filename
        """

        fname = "{}_{}.png".format(index, rythm)
        fpath = os.path.join(dataset_dir, fname)

        x = np.arange(len(signal))

        fig = plt.figure(frameon=False)
        plt.plot(x, signal)
        fig.savefig(fpath)    

        return fname