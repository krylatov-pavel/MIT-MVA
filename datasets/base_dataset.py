class BaseDataset(object):
    def get_input_fn(self, mode, fold_num=None):
        raise NotImplementedError("get_input_fn not implemented")