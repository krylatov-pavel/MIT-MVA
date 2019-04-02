from collections import namedtuple

Record = namedtuple("Record", ["signal", "annotation"])
Slice = namedtuple("Slice", ["record", "rythm", "start", "end", "signal"])
Scale = namedtuple("Scale", ["min", "max"])
Image = namedtuple("Image", ["data", "label", "name"])
CropMode = namedtuple("CropMode", ["vertical", "horizontal"])
Example = namedtuple("Example", ["x", "y"])

class Crop:
    TOP = "top"
    BOTTOM = "bottom"
    LEFT = "left"
    RIGHT = "right"
    CENTER = "center"