class ECG(object):
    def __init__(self, name, signals, annotations):
        self.name = name
        self.signals = signals
        self.annotations = annotations

    def get_samples(self, sample_len, signal_type_filter, signal_type_map):
        for ann in self.annotations:
            if ann["label"] in signal_type_map:
                ann["label"] = signal_type_map[ann["label"]]

        filtered_annotations = [ann for ann in self.annotations if ann["label"] in signal_type_filter]

        #dict, "label" : [[0.2323, 0.1231, ...], ...]
        sample_groups = {} 

        for annotation in filtered_annotations:
            seq_label = annotation["label"]
            seq_start = annotation["start"]
            seq_end = annotation["end"]
            
            if not seq_label in sample_groups:
                sample_groups[seq_label] = []

            seq_len = seq_end - seq_start
            sample_num = seq_len // sample_len
            
            signals = [self.signals[seq_start + (i * sample_len):seq_start + ((i+1) * sample_len)] for i in range(sample_num)]
            sample_groups[seq_label].extend(signals)

        return sample_groups