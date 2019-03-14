import tensorflow as tf
from metrics.base_pos_neg_metric import BasePosNegMetric

class FalseNegatives(BasePosNegMetric):
    def __init__(self, class_name):
        super(FalseNegatives, self).__init__(class_name, "fn")

    def _calc_metric(self, labels, predictions):
        class_mask = tf.equal(labels, self.class_name) # probably you need to conver class_name to scalar tensor

        class_labels = tf.boolean_mask(labels, class_mask)
        class_predictions = tf.boolean_mask(predictions, class_mask)

        fn_vector = tf.not_equal(class_labels, class_predictions)
        fn_num = tf.reduce_sum(tf.cast(fn_vector, tf.float32), axis=0)

        return fn_num