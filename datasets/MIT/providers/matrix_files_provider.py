import numpy as np
import os
from PIL import Image, ImageDraw
from utils.dirs import create_dirs
from datasets.MIT.base.base_file_provider import BaseFileProvider

class MatrixFileProvider(BaseFileProvider):
    def __init__(self):
        super(MatrixFileProvider, self).__init__(".png")

        self.CORRUPDER_DIR = "corrupted"

        #y-axis scale: 1mV takes 10mm
        self.MM_IN_MV = 10
        #x-axis scale: 1sec takes 25mm
        self.MM_IN_SEC = 75

        self.CORRUPDED_TRESHOLD = 0.05

    def _read_file(self, fpath):
        """Read and return file data
        Implement in an inherited class
        returns image data: [H, W, 1], 0 .. 255 int32
        """
        pic = Image.open(fpath)
        pic_data = np.array(pic.getdata()).reshape(pic.size[1], pic.size[0])

        return np.expand_dims(pic_data, axis=2)

    def _build_save_file_fn(self, directory, params):
        """Returns tuple (save_file_fn, dispose_fn) function
        save_file_fn(signal, file_name)
        """
        y_range = params["y_range"]
        image_height = params["image_height"]
        slice_window = params["slice_window"]
        fs = params["fs"]

        corrupted_dir = os.path.join(directory, self.CORRUPDER_DIR)
        create_dirs([corrupted_dir])

        scale = abs(y_range.min - y_range.max)
        dot_size = scale / image_height #calculate size of one pixel (in milivolts)
        min_idx = y_range.min / dot_size

        physical_width = self.MM_IN_SEC * slice_window / fs
        physical_height = self.MM_IN_MV * scale
        image_width = int(image_height * physical_width / physical_height)

        def save_file_fn(signal, fname):
            if self._is_out_of_range(signal, y_range, self.CORRUPDED_TRESHOLD):
                fpath = os.path.join(corrupted_dir, fname)
            else:
                fpath = os.path.join(directory, fname)

            dot_indexes = np.rint(np.array(signal) / dot_size - min_idx).astype(int)
            
            plot = self._plot(dot_indexes, image_height)
            plot = plot.resize((image_width, image_height))

            plot.save(fpath)

        def dispose_fn():
            return None

        return save_file_fn, dispose_fn

    def _plot(self, dots, h):
        plot = Image.fromarray(np.zeros((h, len(dots)), dtype=np.uint8))

        dots_inverted = h - dots

        draw = ImageDraw.Draw(plot)

        for i in range(1, len(dots_inverted)):
            x1 = i - 1
            x2 = i
            y1 = dots_inverted[x1]
            y2 = dots_inverted[x2]

            if y1 >= 0 and y1 < h and y2 >=0 and y2 < h:
                draw.line((x1, y1, x2, y2), width=2, fill=255)

        del draw

        return plot

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