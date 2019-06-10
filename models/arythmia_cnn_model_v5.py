import tensorflow as tf
import math
from models.base_mit_model import BaseMitModel

class ArythmiaCNNModelv5(BaseMitModel):
    def _network_fn(self, features, mode, scope="MITConvNet", model_dir=None):
        """
        Input size: 1440
        """
        training = mode == tf.estimator.ModeKeys.TRAIN
        pre_logits = features

        filters_num = self._get_hparam("filters_num", default_value=64)
        filters_step = self._get_hparam("filters_step", default_value=32)
        kernel_size = self._get_hparam("kernel_size", default_value=7)
        pool_size = self._get_hparam("pool_size", default_value=2)
        use_batchnorm = self._get_hparam("use_batch_norm", default_value=False)

        signal_len = features.shape[1].value
        conv_layer = 0
        pool_layer = 0
        layers_description = []
        total_learnable_params = 0

        while signal_len > 4:
            for i in range(2):
                if signal_len - kernel_size + 1 < 4:
                    kernel_size = kernel_size - 2
                if kernel_size == 1 or signal_len - kernel_size + 1 < 2:
                    break

                conv_layer += 1
                filters_num = filters_num + filters_step * int(not i) * int(conv_layer > 1)
                pre_logits = tf.layers.conv1d(pre_logits,
                    filters=filters_num,
                    kernel_size=kernel_size,
                    strides=1,
                    padding="valid",
                    activation=None,
                    name="conv{}".format(conv_layer)
                )
                signal_len = signal_len - kernel_size + 1

                learnable_params = kernel_size * filters_num
                total_learnable_params += learnable_params
                layers_description.append("conv{} output_shape: {}x{} output_size: {:,} parameters: {:,}".format(
                    conv_layer,
                    signal_len,
                    filters_num,
                    signal_len * filters_num,
                    learnable_params)
                )

                if use_batchnorm:
                    pre_logits = tf.layers.batch_normalization(pre_logits,
                        training=training,
                        name="conv{}_batch_norm".format(conv_layer)
                    )
                
                pre_logits = tf.nn.relu(pre_logits, name="conv{}_activation".format(conv_layer))

            if math.ceil(signal_len / pool_size) > 4:
                pool_layer += 1
                padding = "same" if signal_len % pool_size else "valid"
                pre_logits = tf.layers.max_pooling1d(pre_logits,
                    pool_size=pool_size,
                    strides=pool_size,
                    padding=padding,
                    name="pool{}".format(pool_layer)
                )
                signal_len = math.ceil(signal_len / pool_size)

                layers_description.append("max_pool{} output_shape: {}x{} output_size: {:,} parameters: 0".format(
                    pool_layer,
                    signal_len,
                    filters_num,
                    signal_len * filters_num)
                )

        pool_layer += 1
        pre_logits = tf.layers.average_pooling1d(pre_logits,
            pool_size=signal_len,
            strides=signal_len,
            padding="valid",
            name="pool{}".format(pool_layer)
        )
        layers_description.append("avg_pool{} output_shape: {}x{} output_size: {:,} parameters: 0".format(pool_layer, 1, filters_num, filters_num))

        pre_logits = tf.layers.flatten(pre_logits, name="flatten")
        signal_len = filters_num
         
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
            learnable_params = dense_units * signal_len
            total_learnable_params += learnable_params
            signal_len = dense_units
            layers_description.append("fc{} output_shape: {}x1 output_size: {} parameters: {:,}".format(
                i,
                dense_units,
                dense_units,
                learnable_params)
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

        learnable_params = self._get_hparam("class_num", 4) * signal_len
        total_learnable_params += learnable_params
        layers_description.append("fc{} output_shape: {}x1 output_size: {} parameters: {:,}".format(
            dense_layers + 1,
            self._get_hparam("class_num", 4),
            self._get_hparam("class_num", 4),
            learnable_params)
        )

        self._description = "\n".join(layers_description + ["Total params: {:,}".format(total_learnable_params)])

        return logits            