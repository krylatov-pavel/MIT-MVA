class BaseDataset(object):
    def get_input_fn(self, mode):
        raise NotImplementedError("get_input_fn not implemented")