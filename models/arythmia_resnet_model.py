import tensorflow as tf
import math
from models.base_mit_model import BaseMitModel

class ArythmiaResnetModel(BaseMitModel):
    def __init__(self, hparams, dataset_params):
        super(ArythmiaResnetModel, self).__init__(hparams, dataset_params)

        self.CONV_IN_RES = 2 #number of convolutional layers in res block
        self.RES_IN_GROUP = 2 #number of res blocks before downsampling

        self.kernel_size = self._get_hparam("kernel_size", default_value=0)
        self.use_batchnorm = self._get_hparam("use_batch_norm", default_value=False)
        self.filters_num = self._get_hparam("filters_num", default_value=0)
        self.conv_layers = self._get_hparam("conv_layers", default_value=0)

        if not self.conv_layers % 2:
            raise ValueError("invalid conv_layers number: {}. value must be odd.".format(self.conv_layers))

    def _network_fn(self, features, mode, scope="MITConvNet"):
        training = mode == tf.estimator.ModeKeys.TRAIN
        
        pre_logits = features
        pre_logits = self._conv_block(pre_logits, self.filters_num, training, 0)

        downsample_num = math.ceil((self.conv_layers - 1) / (self.CONV_IN_RES * self.RES_IN_GROUP))
        conv_layers = 1

        for i in range(downsample_num):
            for j in range(self.RES_IN_GROUP):
                if conv_layers < self.conv_layers:
                    pre_logits = self._residual_block(pre_logits,
                        downsample=not j,
                        training=training,
                        i=i * self.RES_IN_GROUP + j)
                    conv_layers += self.CONV_IN_RES

        pre_logits = tf.layers.average_pooling1d(pre_logits,
            pool_size=2,
            strides=2,
            padding="same",
            name="avg_pool_global"
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
                name="dense{}".format(i)
            )

            if self.use_batchnorm:
                pre_logits = tf.layers.batch_normalization(pre_logits,
                    training=training,
                    name="dense{}_batch_norm".format(i)
                )

            if use_dropout:
                pre_logits = tf.layers.dropout(pre_logits,
                   rate=dropout_rate,
                   training=training,
                   name="dense{}_dropout".format(i)
                )
        
        logits = tf.layers.dense(pre_logits,
            units=self._get_hparam("class_num", 4),
            name="dense_logits"
        )

        return logits

    def _conv_block(self, x, filters_num, training, i):
        output = tf.layers.conv1d(x,
                filters=filters_num,
                kernel_size=self.kernel_size,
                strides=1,
                padding="same",
                activation=None,
                name="conv{}".format(i)
            )

        if self.use_batchnorm:
            output = tf.layers.batch_normalization(output,
                training=training,
                name="conv{}_batch_norm".format(i)
            )

        output = tf.nn.relu(output, name="conv{}_activation".format(i))

        return output

    def _residual_block(self, x, downsample, training, i):
        res = x
        filters_num = x.get_shape().as_list()[-1]
        strides = 1

        if downsample:
            filters_num = filters_num * 2
            strides = 2
            res = tf.layers.conv1d(res,
                filters=filters_num,
                kernel_size=1,
                strides=strides,
                padding="same",
                activation=None,
                name="res{}_Ws".format(i)
            )

        output = tf.layers.conv1d(x,
            filters=filters_num,
            kernel_size=self.kernel_size,
            strides=strides,
            padding="same",
            activation=None,
            name="res{}_conv1".format(i)
        )

        if self.use_batchnorm:
            output = tf.layers.batch_normalization(output,
                training=training,
                name="res{}_conv1_batch_norm".format(i)
            )

        output = tf.nn.relu(output, name="res{}_conv1_activation".format(i))

        output = tf.layers.conv1d(output,
            filters=filters_num,
            kernel_size=self.kernel_size,
            strides=1,
            padding="same",
            activation=None,
            name="res{}_conv2".format(i)
        )

        if self.use_batchnorm:
            output = tf.layers.batch_normalization(output,
                training=training,
                name="res{}_conv2_batch_norm".format(i)
            )

        output = tf.nn.relu(output, name="res{}_conv2_activation".format(i))

        output = tf.add(output, res, name="res{}_add".format(i))

        return output    