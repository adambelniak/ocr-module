import tensorflow as tf


def summaries(placeholder_name, scalar_name):
    tf_placeholder = tf.placeholder(tf.float32, shape=None, name=placeholder_name)
    tf_scalar_summary = tf.summary.scalar(scalar_name, tf_placeholder)
    return tf_scalar_summary, tf_placeholder


def write_summaries(sess, performance_summaries, feed_dict, writer, step):
    summ = sess.run(performance_summaries,
                    feed_dict=feed_dict)

    writer.add_summary(summ, step)
