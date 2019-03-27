from collections import namedtuple

Record = namedtuple("Record", ["signal", "annotation"])
Sample = namedtuple("Sample", ["record", "rythm_type", "start", "end", "signal"])
Scale = namedtuple("Scale", ["min", "max"])