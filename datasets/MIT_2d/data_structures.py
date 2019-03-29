from collections import namedtuple

Record = namedtuple("Record", ["signal", "annotation"])
Sample = namedtuple("Sample", ["record", "rythm", "start", "end", "signal"])
Scale = namedtuple("Scale", ["min", "max"])
Image = namedtuple("Image", ["data", "label", "name"])