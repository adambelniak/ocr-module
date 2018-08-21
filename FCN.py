# --------------------------
# USER-SPECIFIED DATA
# --------------------------
import time

import tensorflow as tf
import helper_batch as helper
# Tune these parameters

num_classes = 3
image_shape = (576, 320)
EPOCHS = 2
BATCH_SIZE = 8
DROPOUT = 0.75

# Specify these directory paths

data_dir = './data'
runs_dir = './runs'
training_dir = './training_set_2'
vgg_path = './data/vgg'

# --------------------------
# PLACEHOLDER TENSORS
# --------------------------

correct_label = tf.placeholder(tf.float32, [None, 576, 320, num_classes], name="y")
learning_rate = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)


# --------------------------
# FUNCTIONS
# --------------------------

def load_vgg(image_shape):
    # load the model and weights
    # model = tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)
    #
    # # Get Tensors to be returned from graph
    # graph = tf.get_default_graph()
    # image_input = graph.get_tensor_by_name('image_input:0')
    # print(image_input.shape)
    # keep_prob = graph.get_tensor_by_name('keep_prob:0')
    # layer3 = graph.get_tensor_by_name('layer3_out:0')
    # layer4 = graph.get_tensor_by_name('layer4_out:0')
    # layer7 = graph.get_tensor_by_name('layer7_out:0')

    x = tf.placeholder(tf.float32, shape=[None, image_shape[0], image_shape[1]])
    input_layer = tf.reshape(x, [-1, image_shape[0], image_shape[1], 3])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[3, 3],
        padding="SAME",
        activation=tf.nn.leaky_relu)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[4, 4],
        padding="SAME",
        activation=tf.nn.leaky_relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[4, 4], strides=4)

    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=64,
        kernel_size=[5, 5],
        padding="SAME",
        activation=tf.nn.leaky_relu)

    # conv3_2 = tf.layers.conv2d(
    #     inputs=conv3,
    #     filters=64,
    #     kernel_size=[3, 3],
    #     padding="SAME",
    #     activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)


    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(pool3, keep_prob)

    return input_layer, keep_prob, pool1, pool2, h_fc1_drop


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    # Use a shorter variable name for simplicity
    layer3, layer4, layer7 = vgg_layer3_out, vgg_layer4_out, vgg_layer7_out

    # Apply 1x1 convolution in place of fully connected layer
    fcn8 = tf.layers.conv2d(layer7, filters=num_classes, kernel_size=1, name="fcn8")

    # Upsample fcn8 with size depth=(4096?) to match size of layer 4 so that we can add skip connection with 4th layer


    fcn9 = tf.layers.conv2d_transpose(fcn8, filters=layer4.get_shape().as_list()[-1],
                                      kernel_size=5, strides=(2, 2), padding='SAME', name="fcn9")
    # Add a skip connection between current final layer fcn8 and 4th layer
    fcn9_skip_connected = tf.add(fcn9, layer4, name="fcn9_plus_vgg_layer4")

    # Upsample again
    fcn10 = tf.layers.conv2d_transpose(fcn9_skip_connected, filters=layer3.get_shape().as_list()[-1],
                                       kernel_size=4, strides=(4, 4), padding='SAME', name="fcn10_conv2d")

    # Add skip connection
    fcn10_skip_connected = tf.add(fcn10, layer3, name="fcn10_plus_vgg_layer3")

    # Upsample again
    fcn11 = tf.layers.conv2d_transpose(fcn10_skip_connected, filters=num_classes,
                                       kernel_size=16, strides=(2, 2), padding='SAME', name="fcn11")

    return fcn11


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    # Reshape 4D tensors to 2D, each row represents a pixel, each column a class
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name="fcn_logits")
    correct_label_reshaped = tf.reshape(correct_label, (-1, num_classes))
    print(logits.shape)
    print(correct_label_reshaped.shape)

    # Calculate distance from actual labels using cross entropy
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label_reshaped[:])
    # Take mean for total loss
    loss_op = tf.reduce_mean(cross_entropy, name="fcn_loss")

    # The model implements this operation to find the weights/parameters that would yield correct pixel labels
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op, name="fcn_train_op")

    return logits, train_op, loss_op


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op,
             cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    keep_prob_value = 0.5
    learning_rate_value = 0.001
    print("START")
    for epoch in range(epochs):
        # Create function to get batches
        total_loss = 0
        for i, (X_batch, gt_batch) in enumerate(get_batches_fn(batch_size)):
            print(i)
            loss, _ = sess.run([cross_entropy_loss, train_op],
                               feed_dict={input_image: X_batch, correct_label: gt_batch,
                                          keep_prob: keep_prob_value, learning_rate: learning_rate_value})

            total_loss += loss;

        print("EPOCH {} ...".format(epoch + 1))
        print("Loss = {:.3f}".format(total_loss))
        print()


def run():
    get_batches_fn = helper.gen_batch_function(training_dir, image_shape)
    with tf.Session() as session:
        image_input, keep_prob, layer3, layer4, layer7 = load_vgg(image_shape)

        model_output = layers(layer3, layer4, layer7, num_classes)

        # Returns the output logits, training operation and cost operation to be used
        # - logits: each row represents a pixel, each column a class
        # - train_op: function used to get the right parameters to the model to correctly label the pixels
        # - cross_entropy_loss: function outputting the cost which we are minimizing, lower cost should yield higher accuracy
        logits, train_op, cross_entropy_loss = optimize(model_output, correct_label, learning_rate, num_classes)

        # Initialize all variables
        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())

        print("Model build successful, starting training")
        writer = tf.summary.FileWriter('.')
        writer.add_graph(tf.get_default_graph())
        # Train the neural network
        start_time = time.time()

        train_nn(session, EPOCHS, BATCH_SIZE, get_batches_fn,
                 train_op, cross_entropy_loss, image_input,
                 correct_label, keep_prob, learning_rate)
        elapsed_time = time.time() - start_time

        # Run the model with the test images and save each painted output image (roads painted green)
        helper.save_inference_samples(runs_dir, data_dir, session, image_shape, logits, keep_prob, image_input)
        print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
)
        print("All done!")


# --------------------------
# MAIN
# --------------------------
if __name__ == '__main__':
    run()