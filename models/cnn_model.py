import tensorflow as tf
from models.base_mit_model import BaseMitModel

class CNNModel(BaseMitModel):
    def _network_fn(self, features, mode, scope="MITConvNet"):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            flatten = tf.layers.flatten(features)
            fc1 = tf.layers.dense(flatten, units=self._hparams.class_num)

        return fc1