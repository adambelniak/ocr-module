import os, argparse

import tensorflow as tf

# The original freeze_graph function
# from tensorflow.python.tools.freeze_graph import freeze_graph

dir = os.path.dirname(os.path.realpath(__file__))


def freeze_graph(model_dir):
    """Extract the sub graph defined by the output nodes and convert
    all its variables into constant
    Args:
        model_dir: the root folder containing the checkpoint state file
        output_node_names: a string, containing all the output node's names,
                            comma separated
    """
    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)



    # We retrieve our checkpoint fullpath
    absolute_model_dir = './saved_model'

    output_graph = absolute_model_dir + "/frozen_model.pb"


    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], './saved_model')
        graph = tf.get_default_graph()
        image_shape = (512, 384)

        y = graph.get_tensor_by_name('fcn_logits:0')
        input_layer = tf.reshape(y, [ image_shape[0] * image_shape[1] * 3], name='output')

        # saver = tf.train.import_meta_graph('./saved_model/saved_model.ckpt.meta', clear_devices=True)
        #
        # # We restore the weights
        # saver.restore(sess, './saved_model/variables/variables.data-00000-of-00001')

        # We use a built-in TF helper to export variables to constants
        # output_graph_def = tf.graph_util.convert_variables_to_constants(
        #     sess,  # The session is used to retrieve the weights
        #     tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
        #     [ 'y'] # The output node names are used to select the usefull nodes
        # )
        # writer = tf.summary.FileWriter('.')
        # writer.add_graph(output_graph_def)
        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(tf.get_default_graph().as_graph_def().SerializeToString())
        print("%d ops in the final graph." % len(tf.get_default_graph().as_graph_def().node))

    # return output_graph_def

def frezee_2():
    from tensorflow.python.tools import freeze_graph

    # Freeze the graph
    DESTINATION = './save_class/'
    SAVED_MODEL_PATH = './saved_model_class/'
    MODEL_NAME = 'saved_model'
    # graph definition saved above
    input_graph = SAVED_MODEL_PATH + MODEL_NAME + '.pb'
    # any other saver to use other than default
    input_saver = ""
    # earlier definition file format text or binary
    input_binary = True
    # checkpoint file to merge with graph definition
    input_checkpoint = './saved_model_class/saved_model.ckpt'
    # output nodes inn our model
    output_node_names = 'output'
    restore_op_name = 'frozen_save_class/restore_all'
    filename_tensor_name = 'frozen_save_class/Const:0'
    # output path
    output_graph = DESTINATION + 'frozen_' + MODEL_NAME + '.pb'
    # default True
    clear_devices = True
    initializer_nodes = ""
    variable_names_blacklist = ""

    freeze_graph.freeze_graph(
        input_graph,
        input_saver,
        input_binary,
        input_checkpoint,
        output_node_names,
        restore_op_name,
        filename_tensor_name,
        output_graph,
        clear_devices,
        initializer_nodes,
        variable_names_blacklist
    )

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
        tf.import_graph_def(graph_def, name="prefix")
        with tf.Session() as sess:

            writer = tf.summary.FileWriter('.')
            writer.add_graph(tf.get_default_graph())
    return graph

if __name__ == '__main__':
    # freeze_graph('saved_model')
    frezee_2()
    # load_graph('./save/frozen_ocr.pb')