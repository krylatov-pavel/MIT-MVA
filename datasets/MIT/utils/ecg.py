import numpy as np
from datasets.MIT.utils.data_structures import Slice

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
    
    def get_slices(self, slice_window, rythm_filter, rythm_map, reverse=False):
        """Cuts heart rythm sequences into a set of fixed-length slices
        Args:
            slice_window: int, slice length in frames
            rythm_filter: list of heart rythm types that needs to be included in slices,
            e.g ["(ASYS", "(VT", ...]
            rythm_map: in case some labels have the same meaning, like "(N" and "(NSR" map them to
            the same label for convinience. Dictionary, e.g:
            {
                "(NSR)": "(N)",
                ...
            }
        Returns:
            list of Slice, each slice is a named tuple ("record", "rythm", "start", "end", "signal"), e.g:
            [("(N", 32, 1001), ...]
        """

        slices = []
        
        for label, start, end in zip(self.labels, self.timecodes, np.append(self.timecodes[1:], len(self.signal))):
            if label in rythm_map:
                label = rythm_map[label]
            
            if label in rythm_filter:
                slices.extend(self._cut_slices(slice_window, label, start, end, reverse))

        return slices

    def _cut_slices(self, slice_window, label, start, end, reverse=False):
        """ Cust single heart rythm sequence into fixed-length slices
        Args:
            start: sequence start position, inclusive
            end: sequence end position, exclusive
            reverse: if True, start slicing from the end of a sequence
        """
        slice_num = (end - start) // slice_window
        slices = [None] * slice_num

        for i in range(slice_num):
            if reverse:
                start_pos = start + i * slice_window
                end_pos = start_pos + slice_window
            else:
                end_pos = end - i * slice_window
                start_pos = end_pos - slice_window
            
            signal = list(self.signal[start_pos:end_pos])

            slices[i] = Slice(
                record=self.name,
                rythm=label,
                start=start_pos,
                end=end_pos,
                signal=signal)
        
        return slices