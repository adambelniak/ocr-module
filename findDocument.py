import sys
from time import sleep

import imutils
import tensorflow as tf
import os
import shutil
import numpy as np
import cv2
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import scipy.misc
import math

from tensorflow.core.protobuf import saver_pb2

IMAGE_SHAPE = (64, 48)
BATCH_SIZE = 10
"""
# IT'S IMPORTANT TO SEND TO MODEL IMAGE IN BGR FORMAT

"""


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='VALID')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='VALID')


def prepare_data(dir, data):
    file_names = os.listdir(dir)
    image_paths = []
    for i, file in enumerate(file_names[:600]):
        image_paths.append(os.path.join(dir, file))
        # sys.stdout.flush()
    return image_paths


def build_graph(data, labels):
    x = tf.placeholder(tf.float32, shape=[None, *IMAGE_SHAPE, 3], name='input')
    y_ = tf.placeholder(tf.float32, shape=[None, 2])
    input_layer = tf.reshape(x, [-1, *IMAGE_SHAPE, 3])
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[3, 3],
        padding="SAME",
        activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    # conv2 = tf.layers.conv2d(
    #     inputs=pool1,
    #     filters=32,
    #     kernel_size=[3, 3],
    #     padding="SAME",
    #     activation=tf.nn.relu)
    # pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[4, 4], strides=4)

    # conv3 = tf.layers.conv2d(
    #     inputs=pool2,
    #     filters=64,
    #     kernel_size=[5, 5],
    #     padding="SAME",
    #     activation=tf.nn.relu)
    #
    # pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    # conv4 = tf.layers.conv2d(
    #     inputs=pool3,
    #     filters=96,
    #     kernel_size=[5, 5],
    #     padding="SAME",
    #     activation=tf.nn.relu)
    # conv4_2 = tf.layers.conv2d(
    #     inputs=conv4,
    #     filters=96,
    #     kernel_size=[5, 5],
    #     padding="SAME",
    #     activation=tf.nn.relu)
    #
    # pool4 = tf.layers.max_pooling2d(inputs=conv4_2, pool_size=[2, 2], strides=2)

    pool3_flat = tf.reshape(pool1, [-1, 32 * 24 * 32])
    dense = tf.layers.dense(inputs=pool3_flat, units=512, activation=tf.nn.relu)

    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    h_fc1_drop = tf.nn.dropout(dense, keep_prob)

    logits = tf.layers.dense(inputs=h_fc1_drop, units=2)

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits))
    tf.summary.scalar('cross_entropy', cross_entropy)

    train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    output = tf.nn.softmax(logits, name='output')

    start_session(input_layer, x, output, train_step, y_, keep_prob, accuracy, labels, cross_entropy)


def summaries():
    with tf.name_scope('performance'):
        # Summaries need to display on the Tensorboard
        # Whenever need to record the loss, feed the mean loss to this placeholder
        tf_loss_ph = tf.placeholder(tf.float32, shape=None, name='loss_summary')
        # Create a scalar summary object for the loss so Tensorboard knows how to display it
        tf_loss_summary = tf.summary.scalar('loss', tf_loss_ph)

        # Whenever you need to record the loss, feed the mean test accuracy to this placeholder
        tf_accuracy_ph = tf.placeholder(tf.float32, shape=None, name='accuracy_summary')
        # Create a scalar summary object for the accuracy so Tensorboard knows how to display it
        tf_accuracy_summary = tf.summary.scalar('accuracy', tf_accuracy_ph)
        performance_summaries = tf.summary.merge([tf_loss_summary, tf_accuracy_summary])

        return performance_summaries, tf_loss_ph, tf_accuracy_ph


def start_session(input_layer, x, output, train_step, y_, keep_prob, accuracy , labels, cross_entropy):
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.9

    performance_summaries, tf_loss_ph, tf_accuracy_ph = summaries()
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)
        x_test_images, y_test_labels = get_batches_fn(x_test, y_test, IMAGE_SHAPE)
        writer = tf.summary.FileWriter('.')
        writer.add_graph(tf.get_default_graph())
        summ_writer = tf.summary.FileWriter(os.path.join('summaries', 'first'), sess.graph)
        accuracy_per_epoch = []

        print("Test Data Loaded")
        for i in range(15):
            x_s,  y_s = shuffle(x_train, y_train, random_state=0)

            loss_per_epoch = []
            for bid in range(math.ceil(len(x_s) / BATCH_SIZE)):

                num = len(x_s) - 1 if (bid + 1) * BATCH_SIZE > len(x_s) else (bid + 1) * BATCH_SIZE
                batch = np.array(x_s[bid * BATCH_SIZE:num])
                y_batch = y_s[bid * BATCH_SIZE:num]
                batch, y_batch = get_batches_fn(batch, y_batch, IMAGE_SHAPE)

                print('\r', end='')  # use '\r' to go back
                print(str(bid) + '/' + str(len(x_s) / BATCH_SIZE), end="", flush=True)
                if bid % 40 == 0:
                    train_accuracy = 0
                    for bid_test in range(math.ceil(len(x_test) / BATCH_SIZE)):
                        num = len(x_test_images) - 1 if (bid_test + 1) * BATCH_SIZE > len(x_test_images) else (bid_test + 1) * BATCH_SIZE
                        batch_test = np.array(x_test_images[bid_test * BATCH_SIZE:num])
                        y_batch_test = y_test_labels[bid_test * BATCH_SIZE:num]
                        train_accuracy += accuracy.eval(feed_dict={
                            input_layer: np.array(batch_test), y_: y_batch_test, keep_prob: 1.0})
                    print('step %d, training accuracy %g' % (i, train_accuracy/math.ceil(len(x_test_images) / BATCH_SIZE)))
                    print('step %d, training accuracy %g' % (i, train_accuracy))

                    # summ_writer.add_summary(train_accuracy, i)

                # train_step.run(feed_dict={input_layer: batch, y_: y_batch, keep_prob: 0.5})

                loss, _ = sess.run([cross_entropy, train_step],
                                   feed_dict={input_layer: batch, y_: y_batch,
                                              keep_prob: 0.5})
                loss_per_epoch.append(loss)


            print('Average loss in epoch %d: %.5f' % (i, np.mean(loss_per_epoch)))
            avg_loss = np.mean(loss_per_epoch)
            summ = sess.run(performance_summaries,
                               feed_dict={tf_loss_ph: avg_loss, tf_accuracy_ph: train_accuracy/math.ceil(len(x_test_images) / BATCH_SIZE)})

            # Write the obtained summaries to the file, so it can be displayed in the Tensorboard
            summ_writer.add_summary(summ, i)

        save_model(sess, x, output, keep_prob)


def save_model(sess, x, output, keep_prob):
    try:
        os.mkdir('./saved_model_class')
    except:
        pass

    inputs = {
        "keep_prob": keep_prob,
        "x": x
    }
    outputs = {
        "output": output
    }
    tf.saved_model.simple_save(sess, './simple/saved_model', inputs, outputs)
    tf.get_default_graph()
    saver = tf.train.Saver(write_version=saver_pb2.SaverDef.V2)
    saver.save(sess, "./saved_model_class/saved_model.ckpt")
    tf.train.write_graph(sess.graph_def, './saved_model_class', 'saved_model.pb', as_text=False)


def get_batches_fn(batch, y_batch, image_shape):
    """
    Create batches of training data
    :param batch_size: Batch Size
    :return: Batches of training data
    """
    images = []
    labels = []
    for i, (image_file, label) in enumerate(zip(batch, y_batch)):
        try:
            img = imutils.rotate_bound(scipy.misc.imread(image_file, mode='RGB'), 90)
            if img is not None:
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                img = scipy.misc.imresize(img, image_shape)  / 255
                images.append(img)

                labels.append(label)
                # print("\r import data {:d}/{:d}".format(i, len(batch)), end="")
                # sys.stdout.flush()

        except Exception as e:
            print(e)
            pass
    return np.array(images), np.array(labels)


if __name__ == '__main__':
    with_card = []
    with_card = prepare_data("./sharpness_set/True", with_card)
    without_card = []
    print(np.shape(with_card))
    without_card = prepare_data("./sharpness_set/False", without_card)
    data = with_card + without_card
    y = [[1,0] for x in with_card]
    y += [[0,1] for x in without_card]
    build_graph(data, y)
