import tensorflow as tf
from datasets.base_dataset import BaseDataset

TRAIN = tf.estimator.ModeKeys.TRAIN
EVAL = tf.estimator.ModeKeys.EVAL
PREDICT = tf.estimator.ModeKeys.PREDICT

class BaseMITDataset(BaseDataset):
    def __init__(self, params):
        self.train_batch_size = params["train_batch_size"]
        self.eval_batch_size = params["eval_batch_size"]
        self.slize_window = params["slice_window"]
        self.label_map = params["label_map"]

        self.examples = {}

    def get_eval_examples(self):
        """Used for manual evaluating classifier accuracy
        Returns: tuple
            x: list of eval examples data
            y: list of eval examples labels 
        """
        raise NotImplementedError()       
    
    def _batch_size(self, mode):
        if mode == TRAIN:
            return self.train_batch_size
        else:
            return self.eval_batch_size

    def _folder_numbers(self, mode):
        if mode == TRAIN:
            return [0]
        else:
            return [1]
        
    def _use_augmentated(self, mode):
        if mode == TRAIN:
            #make this True after implement augmentation
            return False
        else:
            return False   