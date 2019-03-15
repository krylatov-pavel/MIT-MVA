import tensorflow as tf
from models.base_mit_model import BaseMitModel

class TwoLayerCNNwBatchNorm(BaseMitModel):
    def _network_fn(self, features, mode):
        training = mode == tf.estimator.ModeKeys.TRAIN

        conv1 = tf.layers.conv1d(features,
            filters=16,
            kernel_size=3,
            strides=1,
            padding="same",
            activation=tf.nn.relu,
            name="conv1"
        )

        batch_norm1 = tf.layers.batch_normalization(conv1,
            training=training,
            name="batch_norm1"
        )

        pool1 = tf.layers.max_pooling1d(batch_norm1,
            pool_size=2,
            strides=2,
            padding="same",
            name="pool1"
        )

        conv2 = tf.layers.conv1d(pool1,
            filters=32,
            kernel_size=3,
            strides=1,
            padding="same",
            name="conv2"
        )

        batch_norm2 = tf.layers.batch_normalization(conv2,
            training=training,
            name="batch_norm2"
        )

        pool2 = tf.layers.max_pooling1d(batch_norm2,
            pool_size=2,
            strides=2,
            padding="same",
            name="pool2"
        )

        flatten = tf.layers.flatten(pool2, name="flatten")

        fc1 = tf.layers.dense(flatten,
            units=self._hparams.class_num,
            name="fc1"
        )
        
        return fc1