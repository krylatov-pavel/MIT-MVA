import tensorflow as tf

dtype = tf.float32

def conv_layer(input, channels_in, channels_out, kernel_size, pad, mode, name="conv"):
    filter_shape = [kernel_size, channels_in, channels_out]
    
    with tf.name_scope(name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            W = tf.get_variable("W", dtype=dtype, initializer=tf.truncated_normal(filter_shape, stddev = 0.1, dtype=dtype))

            z = tf.nn.conv1d(input, W, 1, padding="SAME" if pad else "VALID", name ="z")
            #z = batch_norm_wrapper(z, mode)

            a = tf.nn.relu(z)

            return a

def max_pool(input, pad, name="pool"):
    with tf.name_scope(name):
        return tf.layers.max_pooling1d(input, pool_size=2, strides=2, padding="SAME" if pad else "VALID", name=name)        

def fc_layer(input, size_in, size_out, use_activation, mode, name="fc"):
    with tf.name_scope(name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            W = tf.get_variable("W", dtype=dtype, initializer=tf.truncated_normal([size_in, size_out], stddev = 0.1, dtype=dtype))
            b = tf.get_variable("b", dtype=dtype, initializer=tf.constant(0, shape=[size_out], dtype=dtype))

            z = tf.matmul(input, W) + b
            #z = batch_norm_wrapper(z, mode)

            if use_activation:
                z = tf.nn.relu(z)

            return z