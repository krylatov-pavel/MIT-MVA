import tensorflow as tf
from utils.helpers import unzip_list
from datasets.MIT.base.base_mit_dataset import BaseMITDataset
from datasets.MIT.providers.image_example_provider import ImageExamplesProvider

class ImageDataset(BaseMITDataset):
    def __init__(self, params):
        super(ImageDataset, self).__init__(params, 
            examples_provider=ImageExamplesProvider(params)
        )

    def get_input_fn(self, mode):
        folder_nums = self._folder_numbers(mode)
        use_augmented = self._use_augmented(mode)
        self.examples[mode] = self.examples_provider.get(folder_nums, use_augmented)

        def generator_fn():
            for ex in self.examples[mode]:
                yield (ex.x, self.label_map[ex.y])

        def input_fn():
            image_shape = self.examples[mode][0].x.shape

            dataset = tf.data.Dataset.from_generator(
                generator_fn,
                (tf.float32, tf.int64),
                (tf.TensorShape(image_shape), tf.TensorShape(()))
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

    def _use_augmented(self, mode):
        if mode == tf.estimator.ModeKeys.TRAIN:
            return True
        else:
            return False