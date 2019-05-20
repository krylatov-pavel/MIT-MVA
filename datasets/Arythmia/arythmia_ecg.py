import numpy as np
from datasets.MIT.utils.ecg import ECG

class ArythmiaECG(ECG):
    def __init__(self, name, signal, labels, beats, timecodes):
        super(ArythmiaECG, self).__init__(name, signal, labels, timecodes)
        self.beats = beats

    def get_slices(self, slice_window, rythm_filter, use_augmentation=False):
        slices = []

        curr_rythm = None
        curr_sequence = {
            "name": None,
            "start": None
        }

        for rythm, beat, start, prev in zip(self.labels, self.beats, self.timecodes, np.append(0, self.timecodes[:-1])):
            curr_rythm = rythm if rythm else curr_rythm
            match_name = self._match(rythm_filter, curr_rythm, beat)
            
            #end of a previous rythm sequence
            if curr_sequence["name"] and curr_sequence["name"] != match_name:
                end = prev + ((start - prev) // 2)

                new_slices = self._cut_slices(slice_window, curr_sequence["name"], curr_sequence["start"], end)
                curr_sequence_filter = [f for f in rythm_filter if f.name == curr_sequence["name"]][0]
                if curr_sequence_filter.use_augmentation and len(new_slices) > 0:
                    new_slices.append(self._cut_slices(slice_window, curr_sequence["name"], curr_sequence["start"], end, reverse=True)[0])
                slices.extend(new_slices)

                curr_sequence["name"] = None
                curr_sequence["start"] = None

            #start of a new rythm sequence
            if match_name and match_name != curr_sequence["name"]:
                curr_sequence["name"] = match_name
                curr_sequence["start"] = prev + ((start - prev) // 2)
        
        #if current sequence ends with the ECG
        if curr_sequence["name"]:
            new_slices = self._cut_slices(slice_window, curr_sequence["name"], curr_sequence["start"], len(self.signal))
            slices.extend(new_slices)
        
        return slices

    def _match(self, rythm_filter, rythm, beat):
        for f in rythm_filter:
            if f.rythm:
                if f.rythm != rythm:
                    continue
            if len(f.beats) > 0:
                if not beat in f.beats:
                    continue
            return f.name
        return None