import numpy as np
from datasets.MIT_2d.dataset import Dataset

def test():
    params = {
        "db_name": "vfdb",
        "sample_len": 750,
        "labels_map": {
                "(NSR" : "(N",
                "(VFIB": "(VF"
            },
        "labels_filter": ["(ASYS", "(N", "(VF", "(VT"],
        "split_ratio": [0.8, 0.2]
    }

    ds = Dataset(params)

test()