from collections import namedtuple

Record = namedtuple("Record", ["signal", "annotation"])
Sample = namedtuple("Sample", ["record", "rythm", "start", "end", "signal"])
Scale = namedtuple("Scale", ["min", "max"])
Image = namedtuple("Image", ["data", "label", "name"])
CropMode = namedtuple("CropMode", ["vertical", "horizontal"])

class Crop:
    TOP = "top"
    BOTTOM = "bottom"
    LEFT = "left"
    RIGHT = "right"
    CENTER = "bottom_center"