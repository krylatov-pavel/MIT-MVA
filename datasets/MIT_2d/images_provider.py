import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
from utils.dirs import create_dirs, clear_dir, is_empty

class ImagesProvider(object):
    def __init__(self):
        #y-axis scale: 1mV takes 10mm
        self.MM_IN_MV = 10
        #x-axis scale: 1sec takes 25mm
        self.MM_IN_SEC = 25
        self.CORRUPDER_DIR = "corrupted"

    def load_images(self, directory):
        """Loads images from directory
        Returns:
            images: 4d numpy array, [n, image_width, image_height, channels]
            labels 1d list, e.g ["(N)", "(ASYS", ... ]
        """


    def save_images(self, samples, directory, y_range, sample_len, image_height, fs, dpi=200):
        """Converts sample list to images and saves them to disc
        Args:
            samples: 2d list of samples,
            elements are namedtuples, (Index, rythm, start, end, signal), e.g:
            [[(rythm="(N", start=10, end=760, signal=[0.222, 0.225, ...]), (...)], ...]
            directory: directory to save/load files
            y_range: namedtuple (min, max), denotes voltage range
            image_height: height of saved images, width is calculated
            fs: sample rate, Hz
        Returns:
            None
        """

        if os.path.exists(directory):
            clear_dir(directory)
        else:
            create_dirs([directory])

        corrupted_dir = os.path.join(directory, self.CORRUPDER_DIR)
        create_dirs(corrupted_dir)

        figsize = self._calc_fig_size(image_height=image_height,
            dpi=dpi,
            y_range=y_range,
            sample_len=sample_len,
            fs=fs
        )
        fig = plt.figure(frameon=False, figsize=figsize)

        for sample in samples:
            fname = "{}_{}_{}_{}-{}.png".format(sample.Index, sample.rythm, sample.record, sample.start, sample.end)

            if self._is_out_of_range(sample.signal, y_range, 0.1):
                fpath = os.path.join(directory, self.CORRUPDER_DIR, fname)
            else:
                fpath = os.path.join(directory, fname)

            #hide axis    
            plt.axis('off')

            #hide frame
            plt.box(False)
            
            plt.ylim(y_range.min, y_range.max)

            x = np.arange(len(sample.signal))
            plt.plot(x, sample.signal, linewidth=0.25)
            
            fig.savefig(fpath, dpi=dpi)
            plt.clf() 

            im_gray = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(fpath, im_gray)
        
        plt.close(fig)

    def _calc_fig_size(self, image_height, dpi, y_range, sample_len, fs):
        """Calculate size of output image in inches
        Returns: tuple (width, height)
        """
        duration = sample_len / fs
        width_mm = duration * self.MM_IN_SEC
        height_mm = abs(y_range.max - y_range.min) * self.MM_IN_MV

        height_inch = image_height / dpi
        width_inch = height_inch * width_mm / height_mm

        return (width_inch, height_inch)

    def _is_out_of_range(self, signal, y_range, threshold):
        """Check if part of signal doesn't fit selected signal range
        Args:
            signal: list of p_signal values, e.g [0.22, 0.23, ...]
            y_range: tuple (min, max)
            threshold: float, signal is out of range if more then treshold percentage
                of signal values is out of range
        Returns:
            bool
        """
        out_of_range = [s for s in signal if s < y_range.min or s > y_range.max]
        out_of_range_percentage = len(out_of_range) / len(signal)

        return out_of_range_percentage > threshold