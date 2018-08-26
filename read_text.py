import tensorflow as tf
import os
import scipy.misc
import imutils
import numpy as np

image_shape = (512 , 384)
import cv2
export_dir = './saved_model'
NUMBER_CLASS = 3
from tensorflow.core.protobuf import saver_pb2

def load_model(sess):
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir)
    graph = tf.get_default_graph()
    y = graph.get_tensor_by_name('fcn_logits:0')
    input_layer = tf.reshape(y, [image_shape[0] * image_shape[1] * 3], name='output')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    x = graph.get_tensor_by_name('input:0')
    # input_layer = tf.reshape(x, [-1, image_shape[0], image_shape[1], 3])

    return x, keep_prob, y


def convert_to_meta():
    with tf.Session(graph=tf.Graph()) as sess:
        input_layer, keep_prob, y = load_model(sess)
        meta_graph_def = tf.train.export_meta_graph(filename='./saved_model.meta')
        saver = tf.train.Saver(write_version = saver_pb2.SaverDef.V2)
        save_path = saver.save(sess, "./saved_model/saved_model.ckpt")
        tf.train.write_graph(sess.graph_def, './save', 'ocr' + '.pbtxt')
        tf.train.write_graph(sess.graph_def, './save', 'ocr' + '.pb', as_text=False)

def get_cell(dir_folder, image_path):
    with tf.Session(graph=tf.Graph()) as sess:
        input_layer, keep_prob, y = load_model(sess)

        image = scipy.misc.imresize(
            imutils.rotate_bound(scipy.misc.imread(os.path.join(dir_folder, image_path)), 90), image_shape)
        # TODO Should be moved to tensor
        input = (image - 125) / 255
        print(input.shape)
        im_softmax = sess.run(
            [tf.nn.softmax(y)],
            {keep_prob: 1.0, input_layer: [input]})

        masks = []
        for label in range(0, NUMBER_CLASS):
            segmentation = im_softmax[0][:, label].reshape(image_shape[0], image_shape[1])

            segmentation = (segmentation > 0.33).reshape(image_shape[0], image_shape[1], 1)
            mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
            masks.append(mask)
        return masks, image

def color_cells(data_folder):

    for image_file in os.listdir(data_folder):
        masks, image = get_cell(data_folder, image_file)
        print(np.shape(masks))
        for i, mask in enumerate(masks):
            mask = scipy.misc.toimage(mask, mode="RGBA")
            street_im = scipy.misc.toimage(image)
            street_im.paste(mask, box=None, mask=mask)
            scipy.misc.imsave(os.path.join('./runs', str(i) + image_file), street_im)


if __name__ == '__main__':
    image_path = 'IMG_0369.JPG'
    # color_cells('./test')
    convert_to_meta()