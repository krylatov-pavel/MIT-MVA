import tensorflow as tf
import math
from models.base_model import BaseModel

class BaseMitModel(BaseModel):
    def __init__(self, hparams):
        self._hparams = hparams
    
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
        raise NotImplementedError("Implement this function in inherited class")

    def _make_train_op(self, logits, labels):
        #return loss, optimizer.minimize
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_mean(loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=self._hparams.learning_rate)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

        return loss, train_op