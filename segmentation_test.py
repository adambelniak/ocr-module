import cv2
import imutils
import scipy.misc
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.neighbors.kde import KernelDensity

IMAGE_SHAPE = (512, 384)
TEST_DIRECTORY = 'training_set_500'
MODEL_DIRECTORY = './frozen_cubic/opt_frozen_ocr.pb'


def load_graph(frozen_graph_filename):
    """
    Load tensorflow frozen model to get graph

    :param frozen_graph_filename: str, path to specific saved frozen model
    :return: graph contains input and outputs nodes
    """
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)
    return graph


def get_cell(image_path, sess, graph):
    """
    This function based on trained model, inference masks which are used to image segmentation

    :param image_path: str, path to specific image
    :param sess: tf.Session
    :param graph: tf.Graph
    :return: {array, rgb_image},
    """

    input_layer = graph.get_tensor_by_name('import/input:0')
    bar_code_output = graph.get_tensor_by_name('import/mask_1:0')
    digits_output = graph.get_tensor_by_name('import/mask_2:0')
    thresh = graph.get_tensor_by_name('import/thresh:0')

    keep_prob = graph.get_tensor_by_name('import/keep_prob:0')
    image = imutils.rotate_bound(scipy.misc.imread(image_path), 90)
    image = np.array(image)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    input_image = cv2.resize(image, (IMAGE_SHAPE[1], IMAGE_SHAPE[0]), interpolation=cv2.INTER_NEAREST) / 255

    original = imutils.rotate_bound(cv2.imread(image_path), 90)
    mask_1, mask_2 = sess.run(
        [bar_code_output, digits_output],
        {keep_prob: 1.0, input_layer: [input_image], thresh: [0.3]})

    masks = [mask_1, mask_2]

    return masks, original


def bb_intersection_over_union(predicted_segment, true_mask):
    """
    Count Intersection over Union which is used to evaluate model accuracy

    :return: float, IU accuracy for single test image
    """
    predicted_segment = predicted_segment.reshape(*predicted_segment.shape, 1)
    true_mask = true_mask.reshape(*true_mask.shape, 1)

    full_area = np.any(np.concatenate((predicted_segment, true_mask), axis=2), axis=2)
    inter_area = np.all(np.concatenate((predicted_segment, true_mask), axis=2), axis=2)

    iou = np.sum(inter_area) / np.sum(full_area)
    return iou


def read_true_segments(image_path_folder):
    masks = ['Box11.jpg', 'Box16.jpg']
    masks_image = []
    for mask in masks:
        if os.path.isfile(os.path.join(image_path_folder, mask)):
            masks_image.append(cv2.imread(os.path.join(image_path_folder, mask), cv2.IMREAD_GRAYSCALE))
        else:
            masks_image.append(None)
    return masks_image


def color_cells(image_path_folder, sess, graph):
    try:
        masks, image = get_cell(os.path.join(image_path_folder, 'origin.jpg'), sess, graph)
        true_masks = read_true_segments(image_path_folder)

        iu_accuracy = []
        for i, mask in enumerate(masks):
            open_cv_image = np.array(mask, dtype=np.uint8)
            open_cv_image = np.reshape(open_cv_image, (IMAGE_SHAPE[0], IMAGE_SHAPE[1], 1))
            true_mask = true_masks[i]
            if true_mask is None:
                iu_accuracy.append(None)
                continue
            original_shape = np.shape(true_mask)
            open_cv_image_scaled_up = cv2.resize(open_cv_image, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_AREA)

            iu_accuracy.append(bb_intersection_over_union(open_cv_image_scaled_up, true_mask))
        return iu_accuracy
    except Exception as e:
        print(e)
        pass


def test_model(data_folder, dir_model):
    graph = load_graph(dir_model)
    with tf.Session(graph=graph) as sess:
        iu_accuracy = []
        for image_folder in os.listdir(data_folder)[:10]:
            summ = color_cells(os.path.join(data_folder, image_folder), sess, graph)
            if summ is not None:
                iu_accuracy.append(summ)
        plot_statistics(np.array(iu_accuracy)[:, 0], 'Bar-code')
        plot_statistics(np.array(iu_accuracy)[:, 1], "digits")
        plt.show()
        # iu_average = np.average(iu_accuracy, axis=0)
        # print(iu_average)



def plot_statistics(iu_accuracy, label):
    mask_1_accuracy = list(filter(None.__ne__, iu_accuracy))
    mask_1 = KernelDensity(kernel='epanechnikov', bandwidth= 0.2).fit(np.array(mask_1_accuracy)[:, np.newaxis])
    X_plot = np.linspace(0, 1, 1000)[:, np.newaxis]

    log_dens = mask_1.score_samples(X_plot)
    fig, ax = plt.subplots()
    ax.plot(X_plot[:, 0], np.exp(log_dens), '-',
            label=label)

if __name__ == '__main__':
    test_model(TEST_DIRECTORY, MODEL_DIRECTORY)
