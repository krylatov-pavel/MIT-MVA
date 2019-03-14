import tensorflow as tf

class BasePosNegMetric(object):
    def __init__(self, class_name, metric_name):
        self.class_name = class_name
        self.metric_name = metric_name

    def evaluate(self, labels, predictions):
        with tf.variable_scope("{}_{}".format(self.metric_name, self.class_name)):
            total = tf.get_variable(
                name="total",
                dtype=tf.float32,
                initializer=tf.zeros(()),
                trainable=False,
                collections=[tf.GraphKeys.LOCAL_VARIABLES]
            )

            update_op = tf.assign_add(total, self._calc_metric(labels, predictions))

            return total, update_op
    
    def _calc_metric(self, labels, predictions):
        raise NotImplementedError("implement metric fn in derived class")