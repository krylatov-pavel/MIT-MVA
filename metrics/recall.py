import tensorflow as tf

class Recall(object):
    def __init__(self, class_name):
        self.class_name = class_name

    def evaluate(self, labels, predictions):
        with tf.variable_scope("recall_{}".format(self.class_name)):
            tp_total = tf.get_variable("tp",
                dtype=tf.float32,
                initializer=tf.zeros(()),
                trainable=False,
                collections=[tf.GraphKeys.LOCAL_VARIABLES]
            )

            fn_total = tf.get_variable("fn",
                dtype=tf.float32,
                initializer=tf.zeros(()),
                trainable=False,
                collections=[tf.GraphKeys.LOCAL_VARIABLES]
            )

            update_tp_op = tf.assign_add(tp_total, self._true_positives(labels, predictions))
            update_fn_op = tf.assign_add(fn_total, self._false_negatives(labels, predictions))

            update_op = tf.group([update_tp_op, update_fn_op])

            precision = tp_total / (tp_total + fn_total + tf.keras.backend.epsilon())

            return precision, update_op
    
    def _true_positives(self, labels, predictions):
        class_mask = tf.equal(labels, self.class_name) # probably you need to conver class_name to scalar tensor

        class_labels = tf.boolean_mask(labels, class_mask)
        class_predictions = tf.boolean_mask(predictions, class_mask)

        tp_vector = tf.equal(class_labels, class_predictions)
        tp_num = tf.reduce_sum(tf.cast(tp_vector, tf.float32), axis=0)

        return tp_num

    def _false_negatives(self, labels, predictions):
        class_mask = tf.equal(labels, self.class_name) # probably you need to conver class_name to scalar tensor

        class_labels = tf.boolean_mask(labels, class_mask)
        class_predictions = tf.boolean_mask(predictions, class_mask)

        fn_vector = tf.not_equal(class_labels, class_predictions)
        fn_num = tf.reduce_sum(tf.cast(fn_vector, tf.float32), axis=0)

        return fn_num