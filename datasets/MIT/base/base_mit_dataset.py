import tensorflow as tf
import numpy as np
import pandas as pd
from datasets.base_dataset import BaseDataset
from utils.helpers import unzip_list

TRAIN = tf.estimator.ModeKeys.TRAIN
EVAL = tf.estimator.ModeKeys.EVAL
PREDICT = tf.estimator.ModeKeys.PREDICT

class BaseMITDataset(BaseDataset):
    def __init__(self, params, examples_provider):
        self.train_batch_size = params["train_batch_size"]
        self.eval_batch_size = params["eval_batch_size"]
        self.slize_window = params["slice_window"]
        self.label_map = params["label_map"]
        self.split_ratio = params["split_ratio"]

        self.examples = {}
        self.examples_provider = examples_provider

    def get_eval_examples(self, fold_num=None):
        """Used for manual evaluating classifier accuracy
        Returns: tuple
            x: list of eval examples data
            y: list of eval examples labels 
        """
        EVAL = tf.estimator.ModeKeys.EVAL

        folder_nums = self._folder_numbers(EVAL, fold_num)
        use_augmented = self._use_augmented(EVAL)
        self.examples[EVAL] = self.examples_provider.get(folder_nums, use_augmented)

        return unzip_list(((ex.x, ex.y) for ex in self.examples[EVAL]))
    
    def dataset_stats(self, mode, fold_num=None):
        folder_nums = self._folder_numbers(mode, fold_num)
        use_augmented = self._use_augmented(mode)
        self.examples[mode] = self.examples_provider.get(folder_nums, use_augmented)

        y = [ex.y for ex in self.examples[mode]]
        y_series = pd.Series(y)
        y_distribution = y_series.value_counts(normalize=True).sort_values()

        if fold_num:
            print("fold:", fold_num)    
        print(mode)
        print(y_distribution)
    
    def _batch_size(self, mode):
        if mode == TRAIN:
            return self.train_batch_size
        else:
            return self.eval_batch_size

    def _folder_numbers(self, mode, fold_num=None):
        folds = list(range(len(self.split_ratio)))
        eval_fold = fold_num if fold_num != None else 1

        if mode == TRAIN:
            return [f for f in folds if  f != eval_fold]
        else:
            return [eval_fold]
        
    def _use_augmented(self, mode):
        if mode == TRAIN:
            #make this True after implement augmentation
            return False
        else:
            return False   