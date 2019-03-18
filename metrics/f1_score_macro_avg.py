import tensorflow as tf
from metrics.f1_score import F1Score

class MacroAvgF1Score(object):
    def __init__(self, class_names):
        self.class_names = class_names

    def evaluate(self, labels, predictions):
        with tf.variable_scope("macro_avg_f1_score"):
            class_f1_scores = [F1Score(class_name).evaluate(labels, predictions) for class_name in self.class_names]
            
            f1_update_ops = [f1_score[1] for f1_score in class_f1_scores]
            
            update_op = tf.group(f1_update_ops)
            
            f1_vals = [f1_score[0] for f1_score in class_f1_scores]
            avg_f1_score = tf.reduce_sum(f1_vals, axis=0) / len(self.class_names)

            return avg_f1_score, update_op