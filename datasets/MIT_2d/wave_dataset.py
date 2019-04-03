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
        self.label_map = params["label_map"]

        self.examples_provider = WaveExamplesProvider(params)

    def get_input_fn(self, mode):
        def input_fn():
            dataset = tf.data.Dataset.from_generator(
                self._get_generator_fn(mode),
                (tf.float32, tf.int64),
                (tf.TensorShape((self.slize_window, 1)), tf.TensorShape(()))
            )

            if mode == TRAIN:
                dataset = dataset.shuffle(
                    buffer_size=5000,
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

    def _get_generator_fn(self, mode):
        print("call _get_generator_fn")

        if mode == TRAIN:
            splits = [0]
            #make this True after implement augmentation
            include_aug = False
        else:
            splits = [1]
            include_aug = False

        examples = self.examples_provider.get(splits, include_aug)
        #shuffle(examples)
            
        def generator_fn():
            for ex in examples:
                yield (ex.x, self.label_map[ex.y])

        return generator_fn