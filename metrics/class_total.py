import tensorflow as tf
from metrics.base_pos_neg_metric import BasePosNegMetric

class ClassTotal(BasePosNegMetric):
    def __init__(self, class_name):
        super(ClassTotal, self).__init__(class_name, "total")
    
    def _calc_metric(self, labels, predictions):
        class_mask = tf.equal(labels, self.class_name)

        total = tf.reduce_sum(tf.cast(class_mask, tf.float32), axis=0)

        return total