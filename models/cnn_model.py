import tensorflow as tf
from models.base_mit_model import BaseMitModel

class CNNModel(BaseMitModel):
    def _network_fn(self, features, mode, scope="MITConvNet"):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            conv1 = tf.layers.conv1d(features,
                filters=16,
                kernel_size=3,
                strides=1,
                padding="same",
                activation=tf.nn.relu,
                name="conv1"
            )

            pool1 = tf.layers.max_pooling1d(conv1,
                pool_size=2,
                strides=2,
                padding="same",
                name="pool1"
            )

            flatten = tf.layers.flatten(pool1, name="flatten")

            fc1 = tf.layers.dense(flatten,
                units=self._hparams.class_num,
                name="fc1"
            )
            
            return fc1