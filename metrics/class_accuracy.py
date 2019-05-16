import tensorflow as tf
from metrics.true_positives import TruePositives
from metrics.class_total import ClassTotal

class ClassAccuracy(object):
    def __init__(self, class_label):
        self.class_label = class_label

    def evaluate(self, labels, predictions):
        with tf.variable_scope("accuracy_{}".format(self.class_label)):
            tp, tp_update_op = TruePositives(self.class_label).evaluate(labels, predictions)
            total, total_update_op = ClassTotal(self.class_label).evaluate(labels, predictions)

            update_op = tf.group([tp_update_op, total_update_op])
            accuracy = tp / (total + tf.keras.backend.epsilon())

            return accuracy, update_op