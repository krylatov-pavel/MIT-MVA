import tensorflow as tf
from metrics.true_positives import TruePositives
from metrics.false_negatives import FalseNegatives
from metrics.false_positives import FalsePositives

class MicroAvgF1Score(object):
    def __init__(self, class_names):
        self.class_names = class_names

    def evaluate(self, labels, predictions):
        with tf.variable_scope("macro_avg_f1_score"):
            class_tps = [TruePositives(class_name).evaluate(labels, predictions) for class_name in self.class_names]
            class_fps = [FalsePositives(class_name).evaluate(labels, predictions) for class_name in self.class_names]
            class_fns = [FalseNegatives(class_name).evaluate(labels, predictions) for class_name in self.class_names]

            update_op = tf.group([class_metric[1] for class_metric in class_tps + class_fps + class_fns])

            tp_vals = [tp[0] for tp in class_tps]
            fp_vals = [fp[0] for fp in class_fps] 
            fn_vals = [fn[0] for fn in class_fns]

            tp_sum = tf.reduce_sum(tp_vals, axis=0)

            avg_precision = tp_sum / (tp_sum + tf.reduce_sum(fp_vals, axis=0) + tf.keras.backend.epsilon())
            avg_recall = tp_sum / (tp_sum + tf.reduce_sum(fn_vals, axis=0) + tf.keras.backend.epsilon())

            f1_score = 2 * avg_precision * avg_precision / (avg_precision + avg_recall + tf.keras.backend.epsilon())

            return f1_score, update_op