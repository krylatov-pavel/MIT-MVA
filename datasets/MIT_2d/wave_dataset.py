import tensorflow as tf
from random import shuffle
from datasets.base_dataset import BaseDataset
from datasets.MIT_2d.providers.wave_examples_provider import WaveExamplesProvider

TRAIN = tf.estimator.ModeKeys.TRAIN
EVAL = tf.estimator.ModeKeys.EVAL
PREDICT = tf.estimator.ModeKeys.PREDICT

class WaveDataset(BaseDataset):
    def __init__(self, params):
        self.train_batch_size = params["train_batch_size"]
        self.eval_batch_size = params["eval_batch_size"]
        self.slize_window = params["slice_window"]

        self.examples_provider = WaveExamplesProvider(params)

    def get_input_fn(self, mode):
        def generator_fn():
            if mode == TRAIN:
                splits = [0]
                include_aug = True
            else:
                splits = [1]
                include_aug = False
            
            print("call generator fn")
            examples = self.examples_provider.get(splits, include_aug)
            shuffle(examples)
            
            for ex in examples:
                yield (ex.x, ex.y)
        
        def input_fn():
            dataset = tf.data.Dataset.from_generator(
                generator_fn,
                (tf.float32, tf.int64),
                (tf.TensorShape((self.slize_window, 1)), tf.TensorShape(()))
            )

            if mode == TRAIN:
                dataset = dataset.shuffle(
                    buffer_size=len(5000),
                    reshuffle_each_iteration=True
                ).repeat()
            
            batch_size = self._batch_size(mode)
            dataset = dataset.batch(batch_size)

            return dataset
        
        return input_fn

    def _batch_size(self, mode):
        if mode == TRAIN:
            return self.train_batch_size
        else:
            return self.eval_batch_size    