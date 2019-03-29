import os
import re
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import cv2
from utils.dirs import create_dirs, clear_dir, is_empty
from utils.helpers import unzip_list
from datasets.MIT_2d.data_structures import SampleMetadata

class CropModes:
    TL = "top_left"
    TC = "top_center"
    TR = "top_right"
    CL = "center_left"
    CC = "center"
    CR = "center_right"
    BL = "bottom_left"
    BC = "bottom_center"
    BR = "bottom_right"

class ImagesProvider(object):
    def __init__(self):
        #y-axis scale: 1mV takes 10mm
        self.MM_IN_MV = 10
        #x-axis scale: 1sec takes 25mm
        self.MM_IN_SEC = 25
        self.CORRUPDER_DIR = "corrupted"
        self.AUGMENTED_DIR = "augmented"

    def load_images(self, directory):
        """Loads images from directory
        Returns:
            images: 4d numpy array, [n, image_width, image_height, channels]
            metadata 1d list of SampleMetadata tuples,  e.g [("418", "(N", 10, 760), ...]
        """
        fnames = [f for f in os.listdir(directory) if os.path.isfile(f)]
        images = [None] * len(fnames)
        metadata = [None] * len(fnames)

        for i, fname in enumerate(fnames):
            try:
                img = scipy.misc.imread(fname, flatten=True)
                md = self._get_sample_metadata(fname)
                if md:
                    metadata[i] = md
                    images[i] = img
                else:
                    print("Skipped file {}, can't parse name".format(os.path.join(directory, fname)))
            except:
                print("Skipped file {}, can't read image".format(os.path.join(directory, fname)))
        
        filtered = [(img, md) for img, md in zip(images, metadata) if img != None and md != None]
        images, metadata = unzip_list(filtered)

        return images, metadata

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

    def _get_image_name(self, index, rythm, record, start, end, crop_mode=None):
        postfix = "_{}" + crop_mode if crop_mode else ""
        template = "{}_{}_{}_{}-{}" + postfix + ".png" 
        return template.format(index, rythm, record, start, end)

    def _get_sample_metadata(self, fname):
        regex = "^\d+_(?P<rythm>\(\w+)_(?P<record>\d+)_(?P<start>\d+)-(?P<end>\d+)"
        m = re.match(regex, fname)
        if m:
            return SampleMetadata(
                rythm=m.group('rythm'),
                record=m.group('record'),
                start=m.group('start'),
                end=m.group('end'),
            )
        else:
            return None