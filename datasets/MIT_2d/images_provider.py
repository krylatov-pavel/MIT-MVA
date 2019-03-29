import os
import re
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import cv2
from utils.dirs import create_dirs, clear_dir, is_empty
from datasets.MIT_2d.data_structures import Image, CropMode, Crop

class ImagesProvider(object):
    def __init__(self):
        #y-axis scale: 1mV takes 10mm
        self.MM_IN_MV = 10
        #x-axis scale: 1sec takes 25mm
        self.MM_IN_SEC = 25

        self.CORRUPDER_DIR = "corrupted"
        self.AUGMENTED_DIR = "augmented"
        self.IMG_EXTENSION = ".png"

        self.CROP_RATIO = 0.75

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
            fname = self._generate_img_name(
                index=sample.Index,
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

        aug_map = self._build_augmentation_map(images)

        for i in images:
            for transformation in aug_map[i.label]:
                transformation(i, aug_dir)

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

    def _generate_img_name(self, index, rythm, record, start, end):
        template = "{index}_{rythm}_{record}_{start}-{end}{extension}"
        return template.format(index, rythm, record, start, end, self.IMG_EXTENSION)

    def _generate_aug_img_name(self, original, crop_vertical, crop_horizontal):
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

    def _build_augmentation_map(self, images):
        """Generates mapping of image label to augmentation methods in order to
        equalize image class distribution. i.e apply all available methods to small class,
        several to medium class, few or none to the larges class.
        The same set of transformations is applied to all images with same class 
        Args: 
            images: list of Image tuple (data, label, name)
        Returns:
            dictionary with "label" key and list of augmentation methods, where method
            is a function with "image" argument
            , e.g:
            {
                "(N": [method1, method2, ...]
            }
        """
        aug_map = {}
        img_shape = (images.shape[1], images.shape[2])

        vert_modes = [Crop.TOP, Crop.CENTER, Crop.BOTTOM]
        horiz_modes = [Crop.LEFT, Crop.CENTER, Crop.RIGHT]
        crop_modes = [[CropMode(vert, horiz) for horiz in horiz_modes] for vert in vert_modes]

        labels_series = pd.Series([i.label for i in images])
        labels_distribution = labels_series.value_counts(normalize=True).sort_values()

        min_distribution = labels_distribution.iloc[0] * len(crop_modes)

        for label, distribution in labels_distribution.iteritems():
            aug_num = min_distribution / distribution
            aug_map[label] = [self._build_crop_fn(img_shape, crop_modes[:aug_num])]
        
        return aug_map

    def _build_crop_fn(self, img_shape, crop_modes):
        """Builds function that accepts image as parameter and creates cropped version of this image
        Args:
            img_shape: tuple (width, height), shape of images
            crop_modes: list of tuple CropMode (vertical, horizontal)
        Returns:
            crop function
        """
        w = img_shape[0]
        h = img_shape[1]

        w_crop = int(w * self.CROP_RATIO)
        h_crop = int(h * self.CROP_RATIO)

        top_pads = {
            Crop.TOP: 0,
            Crop.CENTER: int((h - h_crop) / 2),
            Crop.BOTTOM: h - h_crop
        }
        left_pads = {
            Crop.LEFT: 0,
            Crop.CENTER: int((w - self.CROP_RATIO) / 2),
            Crop.RIGHT: w - w_crop
        }

        def crop(image, directory):
            for crop_mode in crop_modes:
                top_pad = top_pads[crop_mode.vertical]
                left_pad = left_pads[crop_mode.horizontal]
                fname = self._generate_aug_img_name(image.name, crop_mode.vertical, crop_mode.horizontal)
                fpath = os.path.join(directory, fname)

                crop = image.data[left_pad:left_pad + w_crop, top_pad:top_pad + h_crop]
                crop = cv2.resize(crop, img_shape)
                cv2.imwrite(fpath, crop)

        return crop