import tensorflow as tf
from models.base_mit_model import BaseMitModel
from utils.tf_utils import normalize_inputs

class CNNModel2d(BaseMitModel):
    def _network_fn(self, features, mode, scope="MITConvNet"):
        training = mode == tf.estimator.ModeKeys.TRAIN
        pre_logits = features

        normalize = self._get_hparam("normalize_inputs", default_value=False)
        if normalize:
            pre_logits = pre_logits / 255

        #L1
        pre_logits = tf.layers.conv2d(pre_logits,
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation=tf.nn.elu,
            name="conv1"
        )

        pre_logits = tf.layers.batch_normalization(pre_logits,
            training=training,
            name="conv1_batch_norm"
        )

        #L2
        pre_logits = tf.layers.conv2d(pre_logits,
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation=tf.nn.elu,
            name="conv2"
        )

        pre_logits = tf.layers.batch_normalization(pre_logits,
            training=training,
            name="conv2_batch_norm"
        )

        #L3
        pre_logits = tf.layers.max_pooling2d(pre_logits,
            pool_size=(2, 2),
            strides=(2, 2),
            padding="same",
            name="pool3"
        )

        #L4
        pre_logits = tf.layers.conv2d(pre_logits,
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation=tf.nn.elu,
            name="conv4"
        )

        pre_logits = tf.layers.batch_normalization(pre_logits,
            training=training,
            name="conv4_batch_norm"
        )

        #L5
        pre_logits = tf.layers.conv2d(pre_logits,
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation=tf.nn.elu,
            name="conv5"
        )

        pre_logits = tf.layers.batch_normalization(pre_logits,
            training=training,
            name="conv5_batch_norm"
        )

        #L6
        pre_logits = tf.layers.max_pooling2d(pre_logits,
            pool_size=(2, 2),
            strides=(2, 2),
            padding="same",
            name="pool6"
        )

        #L7
        pre_logits = tf.layers.conv2d(pre_logits,
            filters=256,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation=tf.nn.elu,
            name="conv7"
        )

        pre_logits = tf.layers.batch_normalization(pre_logits,
            training=training,
            name="conv7_batch_norm"
        )

        #L8
        pre_logits = tf.layers.conv2d(pre_logits,
            filters=256,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation=tf.nn.elu,
            name="conv8"
        )

        pre_logits = tf.layers.batch_normalization(pre_logits,
            training=training,
            name="conv8_batch_norm"
        )

        #L9
        pre_logits = tf.layers.max_pooling2d(pre_logits,
            pool_size=(2, 2),
            strides=(2, 2),
            padding="same",
            name="pool9"
        )

        #L10
        pre_logits = tf.layers.flatten(pre_logits, name="flatten")

        logits = tf.layers.dense(pre_logits,
            units=2048,
            name="dense10",
            activation=tf.nn.elu
        )

        pre_logits = tf.layers.batch_normalization(pre_logits,
            training=training,
            name="dense10_batch_norm"
        )

        pre_logits = tf.layers.dropout(pre_logits,
            rate=0.5,
            training=training,
            name="dense10_dropout"
        )

        #L11
        logits = tf.layers.dense(pre_logits,
            units=self._get_hparam("class_num", 4),
            name="dense_logits"
        )

        return logits            