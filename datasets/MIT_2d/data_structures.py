from collections import namedtuple

Record = namedtuple("Record", ["signal", "annotation"])
Sample = namedtuple("Sample", ["record", "rythm", "start", "end", "signal"])
SampleMetadata = namedtuple("SampleMetadata", ["record", "rythm", "start", "end"])
Scale = namedtuple("Scale", ["min", "max"])