from collections import namedtuple

Record = namedtuple("Record", ["signal", "annotation"])
Sample = namedtuple("Sample", ["rythm_type", "start", "end"])