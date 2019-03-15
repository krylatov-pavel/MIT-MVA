import tensorflow as tf
import math
from models.base_model import BaseModel
from metrics.precision import Precision
from metrics.recall import Recall
from metrics.f1_score_macro_avg import MacroAvgF1Score
from metrics.f1_score_micro_avg import MicroAvgF1Score

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

                class_names = [int(i) for i in range(self._hparams.class_num)]
                
                eval_metric_ops = {
                    "accuracy": accuracy,
                    "macro_avg_f1_score": MacroAvgF1Score(class_names).evaluate(labels, predictions),
                    "micro_avg_f1_score": MicroAvgF1Score(class_names).evaluate(labels, predictions)
                }

                for class_name in class_names:
                    eval_metric_ops["class_precision_{}".format(class_name)] = \
                        Precision(class_name).evaluate(labels, predictions)
                    eval_metric_ops["class_recall_{}".format(class_name)] = \
                        Recall(class_name).evaluate(labels, predictions)

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