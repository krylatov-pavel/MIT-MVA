import tensorflow as tf
from utils.helpers import flatten_list, unzip_list
from datasets.MIT.base.base_mit_dataset import BaseMITDataset
from datasets.MIT.providers.wave_examples_provider import WaveExamplesProvider

class WaveDataset(BaseMITDataset):
    def __init__(self, params):
        super(WaveDataset, self).__init__(params,
            examples_provider=WaveExamplesProvider(params)
        )
        
    def get_input_fn(self, mode):
        folder_nums = self._folder_numbers(mode)
        use_augmented = self._use_augmented(mode)
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

            if mode == tf.estimator.ModeKeys.TRAIN:
                dataset = dataset.shuffle(
                    buffer_size=len(self.examples[mode]),
                    reshuffle_each_iteration=True
                ).repeat()
            
            batch_size = self._batch_size(mode)
            dataset = dataset.batch(batch_size)

            return dataset
        
        return input_fn