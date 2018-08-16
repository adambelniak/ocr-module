import tensorflow as tf
import os
import shutil
import numpy as np
import cv2
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
import random
from sklearn.utils import shuffle

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

def prepare_data(dir, data=[]):
    file_names = os.listdir(dir)
    for file in file_names:
        img = cv2.imread(dir + '/' + file, 0)
        if img is not None:
            res = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
            data.append(res)
    return data


def build_graph(data, labels):
    x = tf.placeholder(tf.float32, shape=[None, 480, 270])
    y_ = tf.placeholder(tf.float32, shape=[None, 2])
    input_layer = tf.reshape(x, [-1, 480, 270, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="valid",
        activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[4, 4], strides=2)

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

    pool3_flat = tf.reshape(pool3, [-1, 54 * 28 * 64])
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
        for i in range(4):
            x_s,  y_s = shuffle(data, labels, random_state=0)
            x_s, x_test = x_s[:-50], x_s[-50:]
            y_s, y_test = y_s[:-50], y_s[-50:]

            writer = tf.summary.FileWriter('.')
            writer.add_graph(tf.get_default_graph())
            for bid in range(int(len(data) / 50)):
                batch = np.array(x_s[bid * 50:(bid + 1) * 50])
                y_batch = y_s[bid * 50:(bid + 1) * 50]
                print(np.array(x_test).shape)

                if bid % 4 == 0:
                    train_accuracy = accuracy.eval(feed_dict={
                        x: np.array(x_test), y_: y_test, keep_prob: 1.0})
                    print('step %d, training accuracy %g' % (i, train_accuracy))
                train_step.run(feed_dict={x: batch, y_: y_batch, keep_prob: 0.5})



if __name__ == '__main__':
    with_card = []
    with_card = prepare_data("./OCR/true", with_card)
    without_card = []
    print(np.shape(with_card))
    without_card = prepare_data("./OCR/false", without_card)
    data = with_card + without_card
    y = [[1,0] for x in with_card]
    y += [[0,1] for x in without_card]
    build_graph(data, y)
