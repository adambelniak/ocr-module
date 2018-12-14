import time
import tensorflow as tf
import fcn_classificator.helper_batch as helper
import os
import utils.performance as performance
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python import debug as tf_debug
from fcn_classificator.metrics import create_metrics_for_one, generate_image

NUM_CLASSES = 3
IMAGE_SHAPE = (512, 384)
OUTPUT_SHAPE = (IMAGE_SHAPE[0] * 1, IMAGE_SHAPE[1] * 1)
EPOCHS = 30
BATCH_SIZE = 8
DROPOUT = 0.5

# Specify these directory paths
runs_dir = './runs'

# --------------------------
# PLACEHOLDER TENSORS
# --------------------------

CORRECT_LABEL = tf.placeholder(tf.float32, [None, *OUTPUT_SHAPE, NUM_CLASSES], name="y")
LEARNING_RATE = tf.placeholder(tf.float32)


# --------------------------
# FUNCTIONS
# --------------------------


def build_cnn_layer(input_layer, filters, kernel_size, pool_size, strides):
    """Build one Neural Network layer

    :param input_layer: output from previous layer
    :param filters:int number of filters in kernel
    :param kernel_size:array contains dim of convolution kernel
    :param pool_size: kernel size of max pooling operation
    :param strides:stride for max pooling
    :return: output from this layer
    """
    conv = tf.layers.conv2d(
        inputs=input_layer,
        filters=filters,
        kernel_size=kernel_size,
        padding="SAME",
        activation=tf.nn.leaky_relu)

    pool = tf.layers.max_pooling2d(inputs=conv, pool_size=pool_size, strides=strides)
    return pool


def build_layer_without_max_pooling(input_layer, filters, kernel_size):
    """Build one Neural Network layer

    :param input_layer: output from previous layer
    :param filters:int number of filters in kernel
    :param kernel_size:array contains dim of convolution kernel
    :param pool_size: kernel size of max pooling operation
    :param strides:stride for max pooling
    :return: output from this layer
    """
    conv = tf.layers.conv2d(
        inputs=input_layer,
        filters=filters,
        kernel_size=kernel_size,
        padding="SAME",
        activation=tf.nn.leaky_relu)
    return conv


def build_convolutional_graph(image_shape):
    """Build first part of the graph to encode input image

    :param image_shape: array
    :return: tf.layers
    """
    x = tf.placeholder(tf.float32, shape=[None, image_shape[0], image_shape[1]])
    input_layer = tf.reshape(x, [-1, image_shape[0], image_shape[1], 3], name='input')

    hidden_1 = build_cnn_layer(input_layer, 32, [3, 3], [2, 2], 2)

    hidden_2 = build_layer_without_max_pooling(hidden_1, 32, [3, 3])
    hidden_2_1 = build_cnn_layer(hidden_2, 64, [3, 3], [2, 2], 2)

    hidden_3 = build_layer_without_max_pooling(hidden_2_1, 96, [5, 5])
    hidden_3_1 = build_cnn_layer(hidden_3, 96, [5, 5], [4, 4], 4)

    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    h_fc1_drop = tf.nn.dropout(hidden_3_1, keep_prob)

    return input_layer, keep_prob, hidden_1, hidden_2_1, h_fc1_drop


def layers(layer1_out, layer2_2_out, layer3_2_out, num_classes):
    """ Build graph which will be decode dat to segmented images
    
    :param layer1_out: 
    :param layer2_2_out: 
    :param layer3_2_out: 
    :param num_classes: 
    :return: last layer in whole model
    """
    # Use a shorter variable name for simplicity
    layer1, layer2, layer3 = layer1_out, layer2_2_out, layer3_2_out

    fcn8 = tf.layers.conv2d(layer3, filters=num_classes, kernel_size=1, name="fcn8")
    fcn9 = tf.layers.conv2d_transpose(fcn8, filters=layer2.get_shape().as_list()[-1],
                                      kernel_size=5, strides=(4, 4), padding='SAME', name="fcn9")

    fcn9_skip_connected = tf.add(fcn9, layer2, name="fcn9_plus_vgg_layer4")
    fcn10 = tf.layers.conv2d_transpose(fcn9_skip_connected, filters=layer1.get_shape().as_list()[-1],
                                       kernel_size=4, strides=(2, 2), padding='SAME', name="fcn10_conv2d")

    fcn10_skip_connected = tf.add(fcn10, layer1, name="fcn10_plus_vgg_layer3")
    fcn11 = tf.layers.conv2d_transpose(fcn10_skip_connected, filters=num_classes,
                                       kernel_size=16, strides=(2, 2), padding='SAME', name="fcn11")
    return fcn11


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """Create optimize function and also calculate accuracy

    :param nn_last_layer: output layer of neural network
    :param correct_label:
    :param learning_rate:float
    :param num_classes:int, number of image segment which we want obtain
    :return: last layers of FCN,
    """
    metrics = create_metrics_for_one(tf.reshape(nn_last_layer, (BATCH_SIZE, -1, num_classes)), tf.reshape(correct_label,
                                         (BATCH_SIZE, -1, num_classes), ), BATCH_SIZE)

    logits = tf.reshape(nn_last_layer, (-1, num_classes), name="fcn_logits")
    correct_label_reshaped = tf.reshape(correct_label, (-1, num_classes))

    # Calculate distance from actual labels using cross entropy
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label_reshaped[:])
    # Take mean for total loss
    loss_op = tf.reduce_mean(cross_entropy, name="fcn_loss")
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op, name="fcn_train_op")
    return logits, train_op, loss_op, metrics


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op,
             cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, images,
             image_shape, output_shape, scalars_metrics, placeholders_metric, metrics_nodes):

    train_writer = tf.summary.FileWriter(os.path.join('train_summaries', 'drugi'), sess.graph)
    learning_rate_value = 0.001
    performance_summaries = tf.summary.merge_all(scope='performance')

    print("START TRAINING")
    for epoch in range(epochs):
        total_loss = 0
        accuracy = []
        IoU = []
        run_options = None
        run_metadata = None

        # if epoch % 10 == 0:
        #     run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        #     run_metadata = tf.RunMetadata()
        #
        for i, (X_batch, gt_batch) in enumerate(get_batches_fn(batch_size)):
            loss, _, acc = sess.run([cross_entropy_loss, train_op, metrics_nodes],
                                    feed_dict={input_image: X_batch, correct_label: gt_batch,
                                               keep_prob: DROPOUT, learning_rate: learning_rate_value},
                                    options=run_options,
                                    run_metadata=run_metadata
                                    )
            print(acc)
            total_loss += loss
            accuracy.append([acc["recall_m_1"], acc["recall_m_2"]])
            IoU.append([acc["iou_m_1"], acc["iou_m_2"]])

        test_accuracy = []
        for i, (X_batch, gt_batch) in enumerate(get_batches_fn(batch_size)):
            test_accuracy.append(
                sess.run([metrics_nodes], feed_dict={input_image: X_batch, correct_label: gt_batch,
                                                     keep_prob: 1.0, }))

            # if i == 0:
            #     img = sess.run(images, feed_dict={input_image: X_batch, correct_label: gt_batch,
            #                                       keep_prob: 1.0, })
            #     train_writer.add_summary(img, 1)

        print('step %d, training accuracy %g' % (i, train_accuracy / math.ceil(len(x_test_images) / BATCH_SIZE)))
        print(test_accuracy)
        if epoch % 10 == 0:
            train_writer.add_run_metadata(run_metadata, 'step%d' % epoch)

        accuracy = np.nan_to_num(np.nanmean(accuracy, axis=0))
        IoU = np.nan_to_num(np.nanmean(IoU, axis=0))
        performance.write_summaries(sess, performance_summaries, {placeholders_metric[0]: accuracy[0], placeholders_metric[1]: accuracy[1], placeholders_metric[2]: IoU[0], placeholders_metric[3]: IoU[1]}, train_writer, epoch)

        print(accuracy)
        print(IoU)

        print("EPOCH {} ...".format(epoch + 1))
        print("Loss = {:.3f}".format(total_loss))


def split_data_set(training_dir, image_shape, output_shape):
    list_path = os.listdir(training_dir)[:80]
    x_train, x_test = train_test_split(list_path, test_size=0.25)

    get_batches_fn = helper.gen_batch_function(training_dir, x_train, image_shape, output_shape)
    get_batches_fn_train = helper.gen_batch_function(training_dir, x_test, image_shape, output_shape)

    return get_batches_fn, get_batches_fn_train


def run(training_dir='../training_set_500'):
    image_shape = IMAGE_SHAPE
    output_shape = OUTPUT_SHAPE
    correct_label = CORRECT_LABEL
    learning_rate = LEARNING_RATE
    get_batches_fn, get_batches_fn_train = split_data_set(training_dir, image_shape, output_shape)
    scalars_metrics = []
    placeholders_metric = []

    sess = tf.Session()
    sess = tf_debug.TensorBoardDebugWrapperSession(sess, 'localhost:6064')

    with tf.Session() as session:

        image_input, keep_prob, layer3, layer4, layer7 = build_convolutional_graph(image_shape)

        model_output = layers(layer3, layer4, layer7, NUM_CLASSES)
        logits, train_op, cross_entropy_loss, metrics_nodes = optimize(model_output, correct_label, learning_rate, NUM_CLASSES)

        with tf.name_scope("performance"):
            for metric in metrics_nodes.keys():
                tf_scalar_summary, tf_placeholder = performance.summaries(metric + "summary", metric)
                scalars_metrics.append(tf_scalar_summary)
                placeholders_metric.append(tf_placeholder)

        images = generate_image(logits, IMAGE_SHAPE)
        images = tf.summary.merge([images])

        # Initiasze all variables
        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())

        print("Model build successful, starting training")
        writer = tf.summary.FileWriter('.')
        writer.add_graph(tf.get_default_graph())
        # Train the neural network
        start_time = time.time()

        train_nn(session, EPOCHS, BATCH_SIZE, get_batches_fn,
                 train_op, cross_entropy_loss, image_input,
                 CORRECT_LABEL, keep_prob, LEARNING_RATE,
                 images, image_shape, output_shape, scalars_metrics, placeholders_metric, metrics_nodes)
        elapsed_time = time.time() - start_time

        inputs = {
            "keep_prob": keep_prob,
            "x": image_input,
        }
        outputs = {"y": logits}
        tf.saved_model.simple_save(
            session,'./saved_model_with_cubic', inputs, outputs
        )

        helper.save_inference_samples(runs_dir, session, image_shape, logits, keep_prob, image_input, output_shape)
        print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
        print("All done!")


if __name__ == '__main__':
    run('123')
