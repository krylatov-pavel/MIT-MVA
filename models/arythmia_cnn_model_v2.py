import tensorflow as tf
from models.base_mit_model import BaseMitModel
from utils.tf_utils import normalize_inputs

class ArythmiaCNNModelv2(BaseMitModel):
    def _network_fn(self, features, mode, scope="MITConvNet"):
        training = mode == tf.estimator.ModeKeys.TRAIN
        pre_logits = features

        conv_layers = self._get_hparam("conv_layers", default_value=0)
        kernel_size = self._get_hparam("kernel_size", default_value=3)
        filters_num = self._get_hparam("filters_num", default_value=16)
        use_batchnorm = self._get_hparam("use_batch_norm", default_value=False)

        for i in range(1, conv_layers + 1):
            filters = filters_num * i
            pre_logits = tf.layers.conv1d(pre_logits,
                filters=filters,
                kernel_size=kernel_size,
                strides=1,
                padding="same",
                activation=None,
                name="conv{}".format(i)
            )

            if use_batchnorm:
                pre_logits = tf.layers.batch_normalization(pre_logits,
                    training=training,
                    name="conv_batch_norm{}".format(i)
                )

            pre_logits = tf.nn.relu(pre_logits, name="activation{}".format(i))

            if not i % 2:
                pre_logits = tf.layers.max_pooling1d(pre_logits,
                    pool_size=2,
                    strides=2,
                    padding="same",
                    name="pool{}".format(i)
                )

        pre_logits = tf.layers.flatten(pre_logits, name="flatten")
         
        dense_layers = self._get_hparam("dense_layers", 0)
        dense_units = self._get_hparam("dense_units", 128)
        use_dropout = self._get_hparam("use_dropout", False)
        dropout_rate= self._get_hparam("dropout_rate", 0.5)


        for i in range(1, dense_layers + 1):
            pre_logits = tf.layers.dense(pre_logits,
                units=dense_units,
                activation=tf.nn.relu,
                name="fc{}".format(i)
            )

            if use_batchnorm:
                pre_logits = tf.layers.batch_normalization(pre_logits,
                    training=training,
                    name="dense_batch_norm{}".format(i)
                )

            if use_dropout:
                pre_logits = tf.layers.dropout(pre_logits,
                   rate=dropout_rate,
                   training=training,
                   name="dense_dropout{}".format(i)
                )
        
        logits = tf.layers.dense(pre_logits,
            units=self._get_hparam("class_num", 4),
            name="dense_logits"
        )

        return logits            