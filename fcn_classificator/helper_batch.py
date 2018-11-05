import cv2
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import time
import tensorflow as tf
import imutils


def gen_batch_function(image_dir, image_paths, image_shape, output_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that git tains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """

        background_color = np.array([255, 255, 255])
        image_path = 'origin.jpg'
        masks_path = ['Box11.jpg', 'Box16.jpg']
        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                try:
                    img = imutils.rotate_bound(scipy.misc.imread(os.path.join(image_dir, image_file, image_path), mode='RGB'), 90)
                    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    if random.random() < 0.5:
                        image = cv2.resize(img, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST) / 255
                    else:
                        image = cv2.resize(img, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_CUBIC) / 255
                    labels = None

                    for mask_path in masks_path:
                        gt_image = None
                        try:
                            gt_image = scipy.misc.imread(os.path.join(image_dir, image_file, mask_path))
                            gt_image = cv2.resize(gt_image, (image_shape[1], image_shape[0]),
                                        interpolation=cv2.INTER_CUBIC)
                        except Exception as e:
                            pass
                        if gt_image is None:
                            gt_bg = np.full(output_shape, False, dtype=bool)
                        else:
                            gt_bg = np.all(gt_image == background_color, axis=2)

                        gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                        if labels is None:
                            labels = gt_bg
                        else:
                            labels = np.concatenate((labels, gt_bg), axis=2)
                    outside_area = np.invert(np.any(labels, axis=2))
                    outside_area = outside_area.reshape(*outside_area.shape, 1)
                    labels = np.concatenate((outside_area, labels), axis=2)
                    gt_images.append(labels)
                    images.append(image)
                except Exception as e:
                    pass

            yield np.array(images), np.array(gt_images)
    return get_batches_fn


def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape, output_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in os.listdir(data_folder):

        img = imutils.rotate_bound(scipy.misc.imread(os.path.join(data_folder, image_file), mode='RGB'), 90)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        if random.random() < 0.5:
            image = cv2.resize(img, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST) / 255
        else:
            image = cv2.resize(img, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_CUBIC) / 255



        output_image = scipy.misc.imresize(
            imutils.rotate_bound(scipy.misc.imread(os.path.join(data_folder, image_file)), 90), output_shape)
        input = image
        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [input]})
        im_softmax = im_softmax[0][:, 1].reshape(output_shape[0], output_shape[1])
        segmentation = (im_softmax > 0.33).reshape(output_shape[0], output_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(output_image)
        street_im.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), np.array(street_im)


def save_inference_samples(runs_dir, sess, image_shape, logits, keep_prob, input_image, output_shape):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, 'test', image_shape, output_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)
