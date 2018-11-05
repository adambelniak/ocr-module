import tensorflow as tf
import numpy as np

def create_metrics_for_one(logits, correct_label_reshaped, batch_size):
    """Calculate IoU metric for each mask and reduce mean

    it means that first accuracy is counted for one image not for whole pixels in batch

    :param logits: output from last layer
    :param correct_label_reshaped:
    :return: mean of accuracy per mask
    """

    softmax = tf.nn.softmax(logits)
    correct_label_reshaped = tf.cast(correct_label_reshaped, tf.int32)
    metrics = {"recall_m_1": [], "recall_m_2": [], "iou_m_1": [], "iou_m_2": []}
    for i in range(1):
        mask_1 = tf.cast(softmax[:, :, 1] > 0.5, tf.int32)
        mask_2 = tf.cast(softmax[:, :, 2] > 0.5, tf.int32)

        metrics["iou_m_1"].append(calculate_IoU_metric(mask_1, correct_label_reshaped[:, :, 1]))
        metrics["iou_m_2"].append(calculate_IoU_metric(mask_2, correct_label_reshaped[:, :, 2]))

        metrics["recall_m_1"].append(calculate_recall_metric(mask_1, correct_label_reshaped[:, :, 1]))
        metrics["recall_m_2"].append(calculate_recall_metric(mask_2, correct_label_reshaped[:, :, 2]))

    for key in metrics.keys():
        mask = tf.greater(tf.stack(metrics[key]), -1.0)
        metrics[key] = tf.reduce_mean(tf.boolean_mask(tf.stack(metrics[key]), mask))

    return metrics


def calculate_IoU_metric(predicted_labels, trn_labels):
    """Calculate IoU metric

    :param predicted_labels: array of boolean,
    :param trn_labels: array of boolean,
    :return: IoU metric = TP / (FP + TP + FN)
    """
    inter = tf.reduce_sum(tf.multiply(predicted_labels, trn_labels))
    union = tf.reduce_sum(tf.subtract(tf.add(predicted_labels, trn_labels), tf.multiply(predicted_labels, trn_labels)))

    IoU = tf.cast(tf.divide(inter, union), tf.float32)

    return tf.cond(tf.equal(tf.reduce_sum(trn_labels), 0), lambda:
            -1.0, lambda: tf.cond(tf.equal(union, 0), lambda: 0.0, lambda: IoU))


def calculate_recall_metric(predicted_labels, trn_labels):
    """Calculate accuracy metric per single mask

    :param predicted_labels: array of boolean,
    :param trn_labels: array of boolean,
    :return: accuracy metric = TP / (TP + FN)
    """
    correctly_predicted = tf.reduce_sum(tf.multiply(predicted_labels, trn_labels))
    true_label = tf.reduce_sum(trn_labels)
    recall = tf.cast(tf.divide(correctly_predicted, true_label), tf.float32)

    return tf.cond(tf.equal(true_label, 0), lambda: -1.0, lambda: recall)



