import os
import re
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.misc
import cv2
from utils.dirs import create_dirs, clear_dir, is_empty
from utils.helpers import flatten_list
from datasets.MIT.utils.data_structures import Example, CropMode, Crop
from datasets.MIT.base.base_file_provider import BaseFileProvider

class ImagesProvider(BaseFileProvider):
    def __init__(self):
        super(ImagesProvider, self).__init__(".png")

        #y-axis scale: 1mV takes 10mm
        self.MM_IN_MV = 10
        #x-axis scale: 1sec takes 25mm
        self.MM_IN_SEC = 25

        self.CORRUPDER_DIR = "corrupted"
        
        self.CORRUPDED_TRESHOLD = 0.05
        self.CROP_RATIO = 0.8

    def augment(self, directory, crop_ratio=0.75):
        """Reads existing images from directory and creates augmented versions, saves them on disk
        Args:
            directory: path to existing dataset images
        """
        aug_dir = os.path.join(directory, self.AUGMENTED_DIR)
        
        if os.path.exists(aug_dir):
            clear_dir(aug_dir)
        else:
            create_dirs([aug_dir])

        images = self._load_dir(directory)

        aug_map = self._build_augmentation_map(images)

        for i in images:
            for transformation in aug_map[i.y]:
                transformation(i, aug_dir)

    def _read_file(self, fpath):
        img = scipy.misc.imread(fpath, flatten=True)
        return np.expand_dims(img, axis=2)

    def _build_save_file_fn(self, directory, params):
        image_height = params["image_height"]
        dpi = params["dpi"]
        y_range = params["y_range"]
        slice_window = params["slice_window"]
        fs = params["fs"]

        corrupted_dir = os.path.join(directory, self.CORRUPDER_DIR)
        create_dirs([corrupted_dir])

        figsize = self._calc_fig_size(image_height=image_height,
            dpi=dpi,
            y_range=y_range,
            slice_window=slice_window,
            fs=fs
        )

        fig = plt.figure(frameon=False, figsize=figsize)
        ax = plt.Axes(fig, [0., 0., 1., 1.])

        def save_file_fn(signal, fname):
            if self._is_out_of_range(signal, y_range, self.CORRUPDED_TRESHOLD):
                fpath = os.path.join(corrupted_dir, fname)
            else:
                fpath = os.path.join(directory, fname)

            ax.set_axis_off()
            fig.add_axes(ax)
            
            plt.ylim(y_range.min, y_range.max)

            x = np.arange(len(signal))
            ax.plot(x, signal, linewidth=0.25)
            
            fig.savefig(fpath, dpi=dpi)
            plt.clf() 

            #convert to grayscale
            im_gray = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(fpath, im_gray)

        def dispose_fn():
            plt.close(fig)

        return save_file_fn, dispose_fn

    def _calc_fig_size(self, image_height, dpi, y_range, slice_window, fs):
        """Calculate size of output image in inches
        Returns: tuple (width, height)
        """
        duration = slice_window / fs
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

    def _build_augmentation_map(self, images):
        """Generates mapping of image label to augmentation methods in order to
        equalize image class distribution. i.e apply all available methods to small class,
        several to medium class, few or none to the larges class.
        The same set of transformations is applied to all images with same class 
        Args: 
            images: list of Example tuple (data, label, name)
        Returns:
            dictionary with "label" key and list of augmentation methods, where method
            is a function with "image" argument
            , e.g:
            {
                "(N": [method1, method2, ...]
            }
        """
        aug_map = {}
        img_shape = (images[0].x.shape[0], images[0].x.shape[1])

        vert_modes = [Crop.TOP, Crop.CENTER, Crop.BOTTOM]
        horiz_modes = [Crop.LEFT, Crop.CENTER, Crop.RIGHT]
        crop_modes = flatten_list([[CropMode(vert, horiz) for horiz in horiz_modes] for vert in vert_modes])

        labels_series = pd.Series([i.y for i in images])
        labels_distribution = labels_series.value_counts(normalize=True).sort_values()

        min_distribution = labels_distribution.iloc[0] * len(crop_modes)

        for label, distribution in labels_distribution.iteritems():
            aug_num = math.ceil(min_distribution / distribution)
            #additional augmentation functions can be added here:
            aug_map[label] = [self._build_crop_fn(img_shape, crop_modes[:aug_num])]
        
        return aug_map

    def _build_crop_fn(self, img_shape, crop_modes):
        """Builds function that accepts image as parameter and creates cropped version of this image
        Args:
            img_shape: tuple (height, width), shape of images
            crop_modes: list of tuple CropMode (vertical, horizontal)
        Returns:
            crop function
        """
        h = img_shape[0]
        w = img_shape[1]

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
                fname = self.name_generator.generate_aug_name(
                    original=image.name,
                    aug_name="{}_{}".format(crop_mode.vertical, crop_mode.horizontal)
                )
                fpath = os.path.join(directory, fname)

                crop = image.x[top_pad:top_pad + h_crop, left_pad:left_pad + w_crop]
                crop = cv2.resize(crop, (w, h))
                cv2.imwrite(fpath, crop)

        return crop