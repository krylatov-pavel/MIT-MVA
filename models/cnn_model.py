import tensorflow as tf
import math
from models.base_model import BaseModel
from utils.tf_helpers import conv_layer, fc_layer, max_pool

class CNNModel(BaseModel):
    def __init__(self, config):
        self.config = config

    def build_model_fn(self):
        def model_fn(features, labels, mode, config):
            logits = self._network_fn(features, mode)
            predictions = tf.argmax(logits, axis=-1)

            loss, train_op = self._make_train_op(logits, labels)

            eval_metric_ops = {
                "accuracy": tf.metrics.accuracy(labels, predictions),
                "precision": tf.metrics.precision(labels, predictions),
                "recall": tf.metrics.recall(labels, predictions)
            }

            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                train_op=train_op,
                eval_metric_ops=eval_metric_ops
            )

        return model_fn

    def _network_fn(self, features, mode, scope="MITConvNet"):
        def calc_output_len(input_len, conv_layers_num, pool_size):
            input = input_len if conv_layers_num == 1 else calc_output_len(input_len, conv_layers_num - 1, pool_size)
            return int(math.ceil(input / pool_size))

        conv1 = conv_layer(features, 1, 16, 3, True, mode, name="conv1")         #[750, 16]
        conv1 = max_pool(conv1, True, "pool1")                  #[375, 16]

        conv2 = conv_layer(conv1, 16, 32, 3, True, mode, name="conv2")    #[375, 32]
        conv2 = max_pool(conv2, True, "pool2")                           #[188, 32]

        input_len = calc_output_len(self.config.sample_len, 2, 2)
        flatten = tf.reshape(conv2, [-1, input_len * 32])

        fc1 = fc_layer(flatten, input_len * 32, 128, True, mode, name="fc1")

        fc2 = fc_layer(fc1, 128, self.config.class_num, False, mode, name="fc2")

        return fc2

    def _make_train_op(self, logits, labels):
        #return loss, optimizer.minimize
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_mean(loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

        return loss, train_op