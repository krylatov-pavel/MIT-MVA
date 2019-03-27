import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from utils.dirs import create_dirs, clear_dir, is_empty

class ImagesProvider(object):
    def convert_to_images(self, sample_groups, dataset_dir, y_scale):
        """Converts sample list to images, saves them to disc
        Args:
            sample_groups: 2d list, length k, where k is number of groups,
            elements are namedtuples, (Index, rythm, start, end, signal), e.g:
            [[(rythm="(N", start=10, end=760, signal=[0.222, 0.225, ...]), (...)], ...]
            dataset_dir: directory to save/load files
            y_scale: namedtuple (min, max), denotes plot scale
        Returns:
            None
        """

        if os.path.exists(dataset_dir) and not is_empty(dataset_dir):
            clear_dir(dataset_dir)

        group_dirs = [os.path.join(dataset_dir, str(i)) for i in range(len(sample_groups))]
        create_dirs(group_dirs)

        fig = plt.figure(frameon=False)

        for i, group in enumerate(sample_groups):
            for sample in group:
                fname = "{}_{}.png".format(sample.Index, sample.rythm)
                fpath = os.path.join(group_dirs[i], fname)

                x = np.arange(len(sample.signal))

                plt.xticks([]), plt.yticks([])
                plt.plot(x, sample.signal)
                plt.ylim(y_scale.min, y_scale.max)
                fig.savefig(fpath)
                plt.clf() 
        
        plt.close(fig)   