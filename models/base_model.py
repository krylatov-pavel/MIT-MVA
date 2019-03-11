class BaseModel(object):
    def build_model_fn(self):
        raise NotImplementedError('build_model_fn not implemented in BaseModel.')