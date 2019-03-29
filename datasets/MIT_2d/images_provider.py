import os
import re
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import cv2
from utils.dirs import create_dirs, clear_dir, is_empty
from datasets.MIT_2d.data_structures import Image

class CropMode:
    TOP = "top"
    BOTTOM = "bottom"
    LEFT = "left"
    RIGHT = "right"
    CENTER = "bottom_center"

class ImagesProvider(object):
    def __init__(self):
        #y-axis scale: 1mV takes 10mm
        self.MM_IN_MV = 10
        #x-axis scale: 1sec takes 25mm
        self.MM_IN_SEC = 25
        self.CORRUPDER_DIR = "corrupted"
        self.AUGMENTED_DIR = "augmented"
        self.IMG_EXTENSION = ".png"

    def load_images(self, directory):
        """Loads images from directory
        Returns:
            list of Image namedtuples (data, label, name),
            where data is 3d numpy array [image_width, image_height, channels]
            label denotes rythm type, eg "(N"
        """
        fnames = [f for f in os.listdir(directory) if os.path.isfile(f)]
        images = [None] * len(fnames)
        labels = [None] * len(fnames)

        for i, fname in enumerate(fnames):
            try:
                img = scipy.misc.imread(fname, flatten=True)
                label = self._get_image_label(fname)
                if label:
                    labels[i] = label
                    images[i] = img
                else:
                    print("Skipped file {}, can't parse name".format(os.path.join(directory, fname)))
            except:
                print("Skipped file {}, can't read image".format(os.path.join(directory, fname)))
        
        filtered = [Image(img, lbl, f) for img, lbl, f in zip(images, labels, fnames) if img != None and lbl != None]

        return filtered

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
            fname = self._generate_image_name(index=sample.Index,
                rythm=sample.rythm,
                record=sample.record,
                start=sample.start,
                end=sample.end
            )

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

            #convert to grayscale
            im_gray = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(fpath, im_gray)
        
        plt.close(fig)

    def augment_images(self, directory, crop_ratio=0.75):
        """Reads existing images from directory and creates augmented versions, saves them on disk
        Args:
            directory: path to existing dataset images
        """
        aug_dir = os.path.join(directory, self.AUGMENTED_DIR)
        
        if os.path.exists(aug_dir):
            clear_dir(aug_dir)
        else:
            create_dirs([aug_dir])

        images = self.load_images(directory)

        w = images.shape[1]
        h = images.shape[2]

        w_crop = int(w * crop_ratio)
        h_crop = int(h * crop_ratio)

        left_pads = [
            (0, CropMode.LEFT),
            (int((w - crop_ratio) / 2), CropMode.CENTER),
            (w - w_crop, CropMode.RIGHT)
        ]
        top_pads = [
            (0, CropMode.TOP),
            (int((h - h_crop) / 2), CropMode.CENTER),
            (h - h_crop, CropMode.BOTTOM)
        ]

        for image in images:
            for left_pad in left_pads:
                for top_pad in top_pads:
                    crop = image[left_pad:left_pad + w_crop, top_pad:top_pad + h_crop]
                    crop = cv2.resize(crop, (w, h))
                    cv2.imwrite(self._generate_augmented_image_name(image.name, top_pad[1], left_pad[1]), crop)

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

    def _generate_image_name(self, index, rythm, record, start, end):
        template = "{index}_{rythm}_{record}_{start}-{end}{extension}"
        return template.format(index, rythm, record, start, end, self.IMG_EXTENSION)

    def _generate_augmented_image_name(self, original, crop_vertical, crop_horizontal):
        return "{name}_{vertical}_{horizontal}{extension}".format(
            original.rstrip(self.IMG_EXTENSION),
            crop_vertical,
            crop_horizontal,
            self.IMG_EXTENSION
        )

    def _get_image_label(self, fname):
        regex = "^\d+_(?P<rythm>\(\w+)_(?P<record>\d+)_(?P<start>\d+)-(?P<end>\d+)"
        m = re.match(regex, fname)
        if m:
            return m.group('rythm')
        else:
            return None