import tensorflow as tf


def summaries(name_scope, placeholder_name, scalar_name):
    with tf.name_scope(name_scope):
        tf_placeholder = tf.placeholder(tf.float32, shape=None, name=placeholder_name)
        tf_scalar_summary = tf.summary.scalar(scalar_name, tf_placeholder)
    return tf_scalar_summary, tf_placeholder
