import tensorflow as tf
from metrics.precision import Precision
from metrics.recall import Recall

class MacroAvgF1Score(object):
    def __init__(self, class_names):
        self.class_names = class_names

    def evaluate(self, labels, predictions):
        with tf.variable_scope("macro_avg_f1_score"):
            class_precisions = [Precision(class_name).evaluate(labels, predictions) for class_name in self.class_names]
            class_recalls = [Recall(class_name).evaluate(labels, predictions) for class_name in self.class_names]
            
            precision_update_ops = [class_precision[1] for class_precision in class_precisions]
            recall_update_ops = [class_recall[1] for class_recall in class_recalls]

            update_op = tf.group(precision_update_ops + recall_update_ops)
            
            precision_vals = [class_precision[0] for class_precision in class_precisions]
            recall_vals = [class_recall[0] for class_recall in class_recalls]

            avg_precision = tf.reduce_sum(precision_vals, axis=0) / len(self.class_names)
            avg_recall = tf.reduce_sum(recall_vals, axis=0) / len(self.class_names)

            f1_score = 2 * avg_precision * avg_recall / (avg_precision + avg_recall + tf.keras.backend.epsilon())

            return f1_score, update_op