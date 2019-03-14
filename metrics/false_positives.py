import tensorflow as tf
from metrics.base_pos_neg_metric import BasePosNegMetric

class FalsePositives(BasePosNegMetric):
    def __init__(self, class_name):
        super(FalsePositives, self).__init__(class_name, "fp")

    def _calc_metric(self, labels, predictions):
            class_mask = tf.equal(predictions, self.class_name)

            class_labels = tf.boolean_mask(labels, class_mask)
            class_predictions = tf.boolean_mask(predictions, class_mask)

            fp_vector = tf.not_equal(class_labels, class_predictions)
            fp_num = tf.reduce_sum(tf.cast(fp_vector, tf.float32), axis=0)

            return fp_num