import tensorflow as tf
from random import shuffle
from datasets.base_dataset import BaseDataset
from datasets.MIT.providers.wave_examples_provider import WaveExamplesProvider

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
        self.examples = {}

    def get_input_fn(self, mode):
        folder_nums = self._folder_numbers(mode)
        use_augmented = self._use_augmentated(mode)
        self.examples[mode] = self.examples_provider.get(folder_nums, use_augmented)

        def generator_fn():
            for ex in self.examples[mode]:
                yield (ex.x, self.label_map[ex.y])

        def input_fn():
            dataset = tf.data.Dataset.from_generator(
                generator_fn,
                (tf.float32, tf.int64),
                (tf.TensorShape((self.slize_window, 1)), tf.TensorShape(()))
            )

            if mode == TRAIN:
                dataset = dataset.shuffle(
                    buffer_size=len(self.examples[mode]),
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