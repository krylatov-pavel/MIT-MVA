import tensorflow as tf
import math
from models.base_model import BaseModel
from metrics.f1_score import F1Score
from metrics.class_accuracy import ClassAccuracy
from metrics.f1_score_macro_avg import MacroAvgF1Score

class BaseMitModel(BaseModel):
    def __init__(self, hparams, dataset_params):
        self._hparams = hparams
        self._label_map = dataset_params["label_map"]
    
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

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            if mode == tf.estimator.ModeKeys.TRAIN:
                if len(update_ops) > 0:
                    train_op = tf.group([train_op, update_ops])

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
                    "macro_avg_f1_score": MacroAvgF1Score(class_names).evaluate(labels, predictions)
                }

                for class_name, class_label in self._label_map.items():
                    eval_metric_ops["accuracy_{}".format(class_name)] = \
                        ClassAccuracy(class_label).evaluate(labels, predictions)

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

    def _get_hparam(self, name, default_value):
        if name in self._hparams:
            return self._hparams[name]
        else:
            return default_value