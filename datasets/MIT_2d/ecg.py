from datasets.MIT_2d.data_structures import Sample

class ECG(object):
    def __init__(self, name, signal, labels, timecodes):
        """
        Args:
            name: record name
            signal: 1d list of physical signal values, e.g [0.233, 0.217, ...]
            labels: 1d list of heart rythm labels, e.g ["(N\x00", "(VT\x00", "(SIN\x00", ...]
            timecodes: 1d list of timecodes corresponding to heart rythm labels, each time code
            denotes starting point of related heart rythm sequence in frames, e.g [34, 2300, 7500, ...]     
        """
        self.name = name
        self.signal = signal
        self.labels = [l.rstrip("\x00") for l in labels] 
        self.timecodes = timecodes
    
    def get_samples(self, sample_len, labels_filter, labels_map):
        """Cuts heart rythm sequences into a set of fixed-length samples
        Args:
            sample_len: int, sample length in frames
            labels_filter: list of herath rythm types that needs to be included in samples,
            e.g ["(ASYS", "(VT", ...]
            labels_map: in case some labels have the same meaning, like "(N" and "(NSR" map them to
            the same label for convinience. Dictionary, e.g:
            {
                "(NSR)": "(N)",
                ...
            }
        Returns:
            list of Samples, each sample is a named tuple (heart_ryth, start, end), e.g:
            [("(N", 32, 1001), ...]
        """

        samples = []
        
        for label, start, end in zip(self.labels, self.timecodes, self.timecodes[1:] + [len(self.signal)]):
            if label in labels_map:
                label = labels_map[label]
            
            if label in labels_filter:
                samples.extend(self._cut_samples(sample_len, label, start, end))

        return samples

    def _cut_samples(self, sample_len, label, start, end):
        """ Cust single heart rythm sequence into fixed-length samples
        Args:
            start: sequence start position, inclusive
            end: sequence end position, exclusive
        """
        sample_num = (end - start) // sample_len
        samples = [None] * sample_num

        for i in range(sample_num):
            start_pos = start + i * sample_len
            end_pos = start_pos + sample_len

            samples[i] = Sample(label, start_pos, end_pos)
        
        return samples