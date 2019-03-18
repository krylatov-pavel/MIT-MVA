import tensorflow as tf
from metrics.precision import Precision
from metrics.recall import Recall

class F1Score(object):
    def __init__(self, class_name):
        self.class_name = class_name

    def evaluate(self, labels, predictions):
        with tf.variable_scope("f1_score_{}".format(self.class_name)):
            precision, precision_update_op = Precision(self.class_name).evaluate(labels, predictions)
            recall, recall_update_op = Recall(self.class_name).evaluate(labels, predictions)

            update_op = tf.group([precision_update_op, recall_update_op])

            f1_score = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())

            return f1_score, update_op