import os, argparse

import tensorflow as tf


dir = os.path.dirname(os.path.realpath(__file__))


def frezee_2():
    from tensorflow.python.tools import freeze_graph

    # Freeze the graph
    DESTINATION = './frozen_cubic/'
    SAVED_MODEL_PATH = './to_froze_save_cubic/'
    MODEL_NAME = 'ocr'
    # graph definition saved above
    input_graph = SAVED_MODEL_PATH + MODEL_NAME + '.pb'
    # any other saver to use other than default
    input_saver = ""
    # earlier definition file format text or binary
    input_binary = True
    # checkpoint file to merge with graph definition
    input_checkpoint = './to_froze_save_cubic/saved_model.ckpt'
    # output nodes inn our model
    output_node_names = 'mask_1,mask_2'
    restore_op_name = 'frozen_cubic/restore_all'
    filename_tensor_name = 'frozen_cubic/Const:0'
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