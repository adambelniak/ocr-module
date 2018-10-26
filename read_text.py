import time

import tensorflow as tf
import os
import scipy.misc
import imutils
import numpy as np
import cv2
from tensorflow.core.protobuf.saved_model_pb2 import SavedModel
from tensorflow.python.saved_model import tag_constants

image_shape = (512, 384)
image_shape_classify = (1024, 768, 3)

export_dir = './saved_model_with_cubic'
NUMBER_CLASS = 3
from tensorflow.core.protobuf import saver_pb2

def load_model(sess):
    # s = SavedModel()

    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir)
    graph = tf.get_default_graph()
    y = graph.get_tensor_by_name('fcn_logits:0')
    out = tf.nn.softmax(y)
    thresh = tf.placeholder(tf.float32, 1, name="thresh")
    mask_1 = tf.to_double(tf.reshape(tf.scalar_mul(255, tf.to_int32(out[:, 1] > thresh),), [image_shape[0] * image_shape[1]]),  name='mask_1')
    mask_2 = tf.to_double(tf.reshape(tf.scalar_mul(255, tf.to_int32(out[:, 2] >thresh),), [image_shape[0] * image_shape[1]]), name='mask_2')
    #
    # keep_prob = graph.get_tensor_by_name('keep_prob:0')
    # x = graph.get_tensor_by_name('input:0')

    writer = tf.summary.FileWriter('.')
    writer.add_graph(graph)

    # input_layer = tf.reshape(x, [-1, image_shape[0], image_shape[1], 3], name='x')
    # writer = tf.summary.FileWriter('.')
    # writer.add_graph(graph)
    # inputs = {
    #     "keep_prob": keep_prob,
    #     "x": x,
    # }
    # outputs = {"y": y}
    # tf.saved_model.simple_save(
    #     sess, './simple_save', inputs, outputs
    # )
    return graph

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def)
    return graph

def convert_to_meta():
    with tf.Session(graph=tf.Graph()) as sess:
        # saver = tf.train.Saver()

        # saver.restore(sess, "/tmp/model.ckpt")
        # new_saver = tf.train.import_meta_graph('saved_model_class/saved_model.ckpt.meta',       clear_devices=True)
        # #
        # new_saver.restore(sess, './saved_model_class/saved_model.ckpt')
        # tf.saved_model.loader.load(sess, [tag_constants.SERVING], './simple_save')

        load_model(sess)
        #
        #
        meta_graph_def = tf.train.export_meta_graph(filename='./saved_model_with_cubic/saved_model.meta')
        saver = tf.train.Saver(write_version = saver_pb2.SaverDef.V2)
        save_path = saver.save(sess, "./to_froze_save_cubic/saved_model.ckpt")
        tf.train.write_graph(sess.graph_def, './to_froze_save_cubic', 'ocr' + '.pbtxt')
        tf.train.write_graph(sess.graph_def, './to_froze_save_cubic', 'ocr' + '.pb', as_text=False)

def get_cell(dir_folder, image_path):
    graph = load_graph('./frozen_cubic/opt_frozen_ocr.pb')

    with tf.Session(graph=graph) as sess:
        # writer = tf.summary.FileWriter('.')
        # writer.add_graph(graph)
        # input_layer, keep_prob, y, y_1 = load_graph(sess)
        input_layer = graph.get_tensor_by_name('import/input:0')
        y = graph.get_tensor_by_name('import/mask_1:0')
        y_1 = graph.get_tensor_by_name('import/mask_2:0')
        thresh = graph.get_tensor_by_name('import/thresh:0')

        keep_prob = graph.get_tensor_by_name('import/keep_prob:0')
        image = imutils.rotate_bound(scipy.misc.imread(os.path.join(dir_folder, image_path)), 90)
        image = np.array(image)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        input = cv2.resize(image, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST) / 255

        cv2.imshow(image_path, input)
        cv2.waitKey(0)
        cv2.imwrite('test.jpg', image)
        cv2.destroyAllWindows()
        original = imutils.rotate_bound(scipy.misc.imread(os.path.join(dir_folder, image_path)), 90)
        # TODO Should be moved to tensor
        print(input.shape)
        mask_1, mask_2 = sess.run(
            [y, y_1],
            {keep_prob: 1.0, input_layer: [input], thresh: [0.1]})

        masks = []
        # im_softmax = np.reshape(im_softmax,(image_shape[0], image_shape[1], 3))
            # segmentation = mask_1.reshape(image_shape[0], image_shape[1])

            # segmentation = (segmentation > 0.33).reshape(image_shape[0], image_shape[1], 1)
        masks.append(mask_1)
        masks.append(mask_2)

        return masks, original


def clasify(data_folder):
    for image_file in os.listdir(data_folder):

        model = load_graph('./frozen_cubic/frozen_ocr2.pb')

        input_layer = model.get_tensor_by_name('import/input:0')
        y = model.get_tensor_by_name('import/output:0')
        keep_prob = model.get_tensor_by_name('import/keep_prob:0')
        try:
            image = scipy.misc.imresize(
                imutils.rotate_bound(scipy.misc.imread(os.path.join(data_folder, image_file)), 90), image_shape_classify)
            scipy.misc.imsave(image_file, image)
            with tf.Session(graph=model) as sess:
                # TODO Should be moved to tensor
                # writer = tf.summary.FileWriter('.')
                # writer.add_graph( tf.get_default_graph())
                input = (image - 125) / 255
                input = np.array(input, dtype=np.float32)
                print(image_file)
                start_time = time.time()
                output = sess.run(
                    [y],
                    {keep_prob: 1.0, input_layer: [input]})
                elapsed_time = time.time() - start_time
                print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
                print(output)
        except:
            pass
def color_cells(data_folder):

    for image_file in os.listdir(data_folder):
        try:
            masks, image = get_cell(data_folder, image_file)
            print(np.shape(masks))
            for i, mask in enumerate(masks):
                open_cv_image = np.array(mask, dtype=np.uint8)
                mask = np.reshape(open_cv_image, (512,384,1))

                # open_cv_image = cv2.resize(open_cv_image, None, fx=8, fy=8)
                # Convert RGB to BGR
                # open_cv_image = open_cv_image[:, :, ::-1].copy()
                # cv2.imshow('cos', mask)
                # _, mask = cv2.threshold(open_cv_image,127,255, cv2.THRESH_BINARY)
                # im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # x, y, w, h = cv2.boundingRect(contours[1])
                # image = np.array(image)
                # box = image[y:y + h, x:x+w]

                cv2.imshow('image', mask)
                cv2.waitKey(0)

                cv2.destroyAllWindows()
                # street_im = scipy.misc.toimage(image)
                # street_im.paste(mask, box=None, mask=mask)
                # scipy.misc.imsave(os.path.join('./runs', str(i) + image_file), street_im)
        except Exception as e:
            print(e)
            pass


if __name__ == '__main__':
    image_path = 'IMG_0369.JPG'
    color_cells('./test')
    # convert_to_meta()
    # clasify('./test')