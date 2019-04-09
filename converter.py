import os
import cv2
import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from utils.dirs import clear_dir, create_dirs
from datasets.MIT_2d.images_provider import ImagesProvider
from datasets.MIT_2d.data_structures import Scale

SOURCE_DIR = "D:\\Study\\MIT\\data\\vfdb\\wave\\750"
TARGET_DIR = "D:\\Study\\MIT\\data\\vfdb\\750_(test_generated_by_converter)"

CORRUPTED_DIR = "corrupted"
AUGMENTED_DIR = "augmented"

SOURCE_EXT = ".csv"
TARGET_EXT = ".png"

CORRUPDED_TRESHOLD = 0.05

#based on collected dataset stats
SIG_MEAN = -0.148
SIG_STD = 1.129

Y_RANGE = Scale(SIG_MEAN - SIG_STD * 2, SIG_MEAN + SIG_STD * 2)
IMAGE_HEIGHT = 128
DPI = 200
SLICE_WINDOW = 750


def convert():
    convert, release = build_converter_fn()

    if os.path.exists(TARGET_DIR):
        clear_dir(TARGET_DIR)

    for d in os.listdir(SOURCE_DIR):
        source_split_dir = os.path.join(SOURCE_DIR, d)
        target_split_dir = os.path.join(TARGET_DIR, d)
        target_corr_dir = os.path.join(target_split_dir, CORRUPTED_DIR)
        target_aug_dir = os.path.join(target_split_dir, CORRUPTED_DIR)

        create_dirs([target_split_dir, target_corr_dir, target_aug_dir])

        source_files = ((os.path.join(source_split_dir, f), f) for f in os.listdir(source_split_dir))
        source_files = (f for f in source_files if os.path.isfile(f[0]))
        
        for fpath, fname in source_files:
            target_fname = fname.rstrip(SOURCE_EXT) + TARGET_EXT
            target_fpath = os.path.join(target_split_dir, target_fname)

            convert(fpath, target_fpath)
    
    release()

def build_converter_fn():
    ip = ImagesProvider()

    figsize = ip._calc_fig_size(image_height=IMAGE_HEIGHT,
            dpi=DPI,
            y_range=Y_RANGE,
            slice_window=SLICE_WINDOW,
            fs=250
        )
    
    fig = plt.figure(frameon=False, figsize=figsize)
    ax = plt.Axes(fig, [0., 0., 1., 1.])

    def convert(source, target):
        with open(source, "r", newline='') as f:
            reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
            signal = list(reader)

        if ip._is_out_of_range(signal, Y_RANGE, CORRUPDED_TRESHOLD):
            target = os.path.join(os.path.dirname(target), CORRUPTED_DIR, os.path.basename(target))

        ax.set_axis_off()
        fig.add_axes(ax)
        
        plt.ylim(Y_RANGE.min, Y_RANGE.max)

        x = np.arange(len(signal))
        ax.plot(x, signal, linewidth=0.25)
        
        fig.savefig(target, dpi=DPI)
        plt.clf() 

        #convert to grayscale
        im_gray = cv2.imread(target, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(target, im_gray)

    def release():
        plt.close(fig)

    return convert, release

convert()