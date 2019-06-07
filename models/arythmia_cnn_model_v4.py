import tensorflow as tf
from models.base_mit_model import BaseMitModel

class ArythmiaCNNModelv4(BaseMitModel):
    def _network_fn(self, features, mode, scope="MITConvNet"):
        """
        Input size: 1456
        Reduce kernel size by 2 every conv layer, down to 3
        """
        training = mode == tf.estimator.ModeKeys.TRAIN
        pre_logits = features

        filters_num = self._get_hparam("filters_num", default_value=64)
        filters_step = self._get_hparam("filters_step", default_value=32)
        use_batchnorm = self._get_hparam("use_batch_norm", default_value=False)

        #inputs 1456x1
        pre_logits = tf.layers.conv1d(pre_logits,
            filters=filters_num,
            kernel_size=9,
            strides=1,
            padding="valid",
            activation=None,
            name="conv{}".format(1)
        )

        if use_batchnorm:
            pre_logits = tf.layers.batch_normalization(pre_logits,
                training=training,
                name="conv{}_batch_norm".format(1)
            )
        
        pre_logits = tf.nn.relu(pre_logits, name="conv{}_activation".format(1))
        
        #inputs 1448x64
        pre_logits = tf.layers.conv1d(pre_logits,
            filters=filters_num,
            kernel_size=9,
            strides=1,
            padding="valid",
            activation=None,
            name="conv{}".format(2)
        )

        if use_batchnorm:
            pre_logits = tf.layers.batch_normalization(pre_logits,
                training=training,
                name="conv{}_batch_norm".format(2)
            )

        pre_logits = tf.nn.relu(pre_logits, name="conv{}_activation".format(2))

        #inputs 1440x64
        pre_logits = tf.layers.max_pooling1d(pre_logits,
            pool_size=3,
            strides=3,
            padding="valid",
            name="pool{}".format(1)
        )

        #inputs 480x64
        filters_num = filters_num + filters_step
        pre_logits = tf.layers.conv1d(pre_logits,
            filters=filters_num,
            kernel_size=7,
            strides=1,
            padding="valid",
            activation=None,
            name="conv{}".format(3)
        )

        if use_batchnorm:
            pre_logits = tf.layers.batch_normalization(pre_logits,
                training=training,
                name="conv{}_batch_norm".format(3)
            )

        pre_logits = tf.nn.relu(pre_logits, name="conv{}_activation".format(3))

        #inputs 474x96
        pre_logits = tf.layers.max_pooling1d(pre_logits,
            pool_size=3,
            strides=3,
            padding="valid",
            name="pool{}".format(2)
        )

        #inputs 158x96
        filters_num = filters_num + filters_step
        pre_logits = tf.layers.conv1d(pre_logits,
            filters=filters_num,
            kernel_size=5,
            strides=1,
            padding="valid",
            activation=None,
            name="conv{}".format(4)
        )

        if use_batchnorm:
            pre_logits = tf.layers.batch_normalization(pre_logits,
                training=training,
                name="conv{}_batch_norm".format(4)
            )

        pre_logits = tf.nn.relu(pre_logits, name="conv{}_activation".format(4))

        #inputs 154x128
        pre_logits = tf.layers.max_pooling1d(pre_logits,
            pool_size=3,
            strides=3,
            padding="same",
            name="pool{}".format(3)
        )

        #inputs 52x128
        filters_num = filters_num + filters_step
        pre_logits = tf.layers.conv1d(pre_logits,
            filters=filters_num,
            kernel_size=5,
            strides=1,
            padding="valid",
            activation=None,
            name="conv{}".format(5)
        )

        if use_batchnorm:
            pre_logits = tf.layers.batch_normalization(pre_logits,
                training=training,
                name="conv{}_batch_norm".format(5)
            )

        pre_logits = tf.nn.relu(pre_logits, name="conv{}_activation".format(5))

        #inputs 48x160
        pre_logits = tf.layers.max_pooling1d(pre_logits,
            pool_size=3,
            strides=3,
            padding="valid",
            name="pool{}".format(4)
        )

        #inputs 16x160
        filters_num = filters_num + filters_step
        pre_logits = tf.layers.conv1d(pre_logits,
            filters=filters_num,
            kernel_size=5,
            strides=1,
            padding="valid",
            activation=None,
            name="conv{}".format(6)
        )

        if use_batchnorm:
            pre_logits = tf.layers.batch_normalization(pre_logits,
                training=training,
                name="conv{}_batch_norm".format(6)
            )

        pre_logits = tf.nn.relu(pre_logits, name="conv{}_activation".format(6))

        #inputs 12x192
        pre_logits = tf.layers.max_pooling1d(pre_logits,
            pool_size=2,
            strides=2,
            padding="valid",
            name="pool{}".format(5)
        )

        #inputs 6x192
        filters_num = filters_num + filters_step
        pre_logits = tf.layers.conv1d(pre_logits,
            filters=filters_num,
            kernel_size=3,
            strides=1,
            padding="valid",
            activation=None,
            name="conv{}".format(7)
        )

        if use_batchnorm:
            pre_logits = tf.layers.batch_normalization(pre_logits,
                training=training,
                name="conv{}_batch_norm".format(7)
            )

        pre_logits = tf.nn.relu(pre_logits, name="conv{}_activation".format(7))

        #inputs 4x224
        pre_logits = tf.layers.max_pooling1d(pre_logits,
            pool_size=4,
            strides=4,
            padding="valid",
            name="pool{}".format(6)
        )

        pre_logits = pre_logits = tf.layers.flatten(pre_logits, name="flatten")
         
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