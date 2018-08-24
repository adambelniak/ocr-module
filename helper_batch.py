import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm
import imutils

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))


def gen_batch_function(data_folder, image_shape, output_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths = os.listdir(data_folder)

        background_color = np.array([255, 255, 255])
        image_path = 'origin.jpg'
        masks_path = ['Box1.jpg', 'Box2.jpg']
        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                try:
                    image = (scipy.misc.imresize(imutils.rotate_bound(scipy.misc.imread(os.path.join(data_folder, image_file, image_path)), 90), image_shape) - 125) / 255
                    labels = None

                    for mask_path in masks_path:
                        gt_image = None
                        try:
                            gt_image = scipy.misc.imresize(scipy.misc.imread(os.path.join(data_folder, image_file, mask_path)), output_shape)
                        except Exception as e:
                            print(e)
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
                    print(e)
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
        image = scipy.misc.imresize(
            imutils.rotate_bound(scipy.misc.imread(os.path.join(data_folder, image_file)), 90), image_shape)
        output_image = scipy.misc.imresize(
            imutils.rotate_bound(scipy.misc.imread(os.path.join(data_folder, image_file)), 90), output_shape)
        input = (image - 125) / 255
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


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image, output_shape):
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
