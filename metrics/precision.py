import tensorflow as tf
from metrics.true_positives import TruePositives
from metrics.false_positives import FalsePositives

class Precision(object):
    def __init__(self, class_name):
        self.class_name = class_name

    def evaluate(self, labels, predictions):
        with tf.variable_scope("precision_{}".format(self.class_name)):
            tp, tp_update_op = TruePositives(self.class_name).evaluate(labels, predictions)
            fp, fp_update_op = FalsePositives(self.class_name).evaluate(labels, predictions)

            update_op = tf.group([tp_update_op, fp_update_op])

            precision = tp / (tp + fp + tf.keras.backend.epsilon())

            return precision, update_op    