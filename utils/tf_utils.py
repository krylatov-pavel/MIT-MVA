import tensorflow as tf

def z_score(inputs, mean, variance):
    return (inputs - mean) / variance

def noramlize_inputs(inputs, mode, decay=0.999):
    
    with tf.variable_scope("inputs_norm"):
        mean = tf.get_variable("mean",
            initializer=tf.zeros([inputs.get_shape()[-1]]),
            trainable=False
        )
        var = tf.get_variable("variance",
            initializer=tf.zeros([inputs.get_shape()[-1]]),
            trainable=False
        )

        if mode == tf.estimator.ModeKeys.TRAIN:
            batch_mean, batch_var = tf.nn.moments(inputs, axes=[0, 1])

            print("batch_mean shape: ", batch_mean.get_shape())
            print("batch_var shape: ", batch_var.get_shape())

            update_mean_op = tf.assign(mean, mean * decay + batch_mean * (1 - decay))
            update_var_op = tf.assign(var, var * decay + batch_var * (1 - decay))

            with tf.control_dependencies([update_mean_op, update_var_op]):
                return z_score(inputs, mean, var)
        else:
            return z_score(inputs, mean, var)

