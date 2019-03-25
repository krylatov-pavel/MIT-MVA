import numpy as np
from datasets.MIT_2d.database_provider import DatabaseProvider
from datasets.MIT_2d.ecg import ECG

def test():
    provider = DatabaseProvider("vfdb")
    records = provider.ged_records()
    record = records[0]
    
    ecg = ECG(record.signal.record_name,
        np.reshape(record.signal.p_signal, [-1]),
        record.annotation.aux_note,
        record.annotation.sample)

    for sample in ecg.get_samples(750, ["(ASYS", "(N", "(VF", "(VT"], {
                "(NSR" : "(N",
                "(VFIB": "(VF"
            }):
        print("Label: {}, start: {}, end: {}, len: {}".format(
                sample.rythm_type, sample.start, sample.end, sample.end - sample.start))

test()