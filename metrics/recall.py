import tensorflow as tf
from metrics.true_positives import TruePositives
from metrics.false_negatives import FalseNegatives

class Recall(object):
    def __init__(self, class_name):
        self.class_name = class_name

    def evaluate(self, labels, predictions):
        with tf.variable_scope("recall_{}".format(self.class_name)):
            tp, tp_update_op = TruePositives(self.class_name).evaluate(labels, predictions)
            fn, fn_update_op = FalseNegatives(self.class_name).evaluate(labels, predictions)
                
            update_op = tf.group([tp_update_op, fn_update_op])

            precision = tp / (tp + fn + tf.keras.backend.epsilon())

            return precision, update_op   