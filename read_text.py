import time

import tensorflow as tf
import helper_batch as helper
import os
import scipy.misc
import imutils
import numpy as np

image_shape = (576, 320)
export_dir = './saved_model'
with tf.Session(graph=tf.Graph()) as sess:
    model = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir)
    graph = tf.get_default_graph()
    y = graph.get_tensor_by_name('fcn_logits:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    x = graph.get_tensor_by_name('input:0')
    input_layer = tf.reshape(x, [-1, image_shape[0], image_shape[1], 3])
    print(x.shape)

    for image_file in os.listdir('./test'):
        image = scipy.misc.imresize(
            imutils.rotate_bound(scipy.misc.imread(os.path.join('./test', image_file)), 0), image_shape)
        input = (image - 125) / 255

        im_softmax = sess.run(
            [tf.nn.softmax(y)],
            {keep_prob: 1.0, x: [input]})

        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        scipy.misc.imsave(os.path.join('./runs', image_file), np.array(street_im))

