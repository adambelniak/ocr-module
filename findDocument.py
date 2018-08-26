import sys

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

IMAGE_SHAPE = (512, 384)
BATCH_SIZE = 20


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
    for i, file in enumerate(file_names):
        image_paths.append(os.path.join(dir, file))
        # sys.stdout.flush()
    return image_paths


def build_graph(data, labels):
    x = tf.placeholder(tf.float32, shape=[None, *IMAGE_SHAPE])
    y_ = tf.placeholder(tf.float32, shape=[None, 2])
    input_layer = tf.reshape(x, [-1, *IMAGE_SHAPE, 3])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[3, 3],
        padding="valid",
        activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="valid",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=64,
        kernel_size=[7, 7],
        padding="valid",
        activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[4, 4], strides=2)

    pool3_flat = tf.reshape(pool3, [-1, 58 * 42 * 64])
    dense = tf.layers.dense(inputs=pool3_flat, units=1024, activation=tf.nn.relu)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(dense, keep_prob)

    logits = tf.layers.dense(inputs=h_fc1_drop, units=2)

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)
        x_test_images, y_test_labels = get_batches_fn(x_test, y_test, IMAGE_SHAPE)
        print("Test Data Loaded")
        for i in range(4):
            x_s,  y_s = shuffle(x_train, y_train, random_state=0)

            writer = tf.summary.FileWriter('.')
            writer.add_graph(tf.get_default_graph())
            for bid in range(math.ceil(len(x_s) / BATCH_SIZE)):
                try:
                    num = len(x_s) - 1 if (bid + 1) * BATCH_SIZE > len(x_s) else (bid + 1) * BATCH_SIZE
                    batch = np.array(x_s[bid * BATCH_SIZE:num])
                    y_batch = y_s[bid * BATCH_SIZE:num]
                    batch, y_batch = get_batches_fn(batch, y_batch, IMAGE_SHAPE)

                    print('\r', end='')  # use '\r' to go back
                    print(str(bid) + '/' + str(len(x_s) / BATCH_SIZE), end="", flush=True)
                    if bid % 10 == 0:
                        train_accuracy = accuracy.eval(feed_dict={
                            input_layer: np.array(x_test_images), y_: y_test_labels, keep_prob: 1.0})
                        print('step %d, training accuracy %g' % (i, train_accuracy))
                    train_step.run(feed_dict={input_layer: batch, y_: y_batch, keep_prob: 0.5})
                except Exception as e:
                    print(e)
                    pass


def get_batches_fn(batch, y_batch, image_shape):
    """
    Create batches of training data
    :param batch_size: Batch Size
    :return: Batches of training data
    """
    images = []
    labels = []
    for i, (image_file, label) in enumerate(zip(batch, y_batch)):
        print('\r', end='')  # use '\r' to go back
        print(str(i), end="", flush=True)
        try:
            img = scipy.misc.imread(image_file)
            if img is not None:
                data.append((scipy.misc.imresize(img, IMAGE_SHAPE) - 125) / 255)
                labels.append(label)
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
