import tensorflow as tf
import math
from models.base_model import BaseModel
from metrics.precision import Precision

class BaseMitModel(BaseModel):
    def __init__(self, hparams):
        self._hparams = hparams
    
    def build_model_fn(self):
        def model_fn(features, labels, mode, config):
            logits = self._network_fn(features, mode)
            predictions = tf.argmax(logits, axis=-1)

            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions = {
                    'class_ids': predictions[:, tf.newaxis],
                    'probabilities': tf.nn.softmax(logits),
                    'logits': logits,
                }

                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions=predictions
                )

            loss, train_op = self._make_train_op(logits, labels)

            if mode == tf.estimator.ModeKeys.TRAIN:
                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=loss,
                    train_op=train_op
                )

            elif mode == tf.estimator.ModeKeys.EVAL:
                accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions)
                precision = tf.metrics.precision(labels=labels, predictions=predictions)
                recall = tf.metrics.recall(labels=labels, predictions=predictions)

                tf.summary.scalar('my_accuracy', accuracy)
                tf.summary.scalar('my_precision', precision)
                tf.summary.scalar('my_recall', recall)

                eval_metric_ops = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "class_precision_0": Precision(0).evaluate(labels, predictions),
                    "class_precision_1": Precision(1).evaluate(labels, predictions),
                    "class_precision_2": Precision(2).evaluate(labels, predictions),
                    "class_precision_3": Precision(3).evaluate(labels, predictions)
                }

                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=loss,
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