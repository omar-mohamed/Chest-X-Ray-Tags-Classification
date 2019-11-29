from __future__ import absolute_import, division

from skimage.transform import resize
from tensorflow.keras.models import model_from_json
import os
import numpy as np
from tensorflow.keras import backend as K
import importlib
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, average_precision_score, accuracy_score, \
    hamming_loss


def set_gpu_usage(gpu_memory_fraction):
    pass
    # if gpu_memory_fraction <= 1 and gpu_memory_fraction > 0:
    #     config = tf.ConfigProto(allow_soft_placement=True)
    #     config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
    #     sess = tf.Session(config=config)
    # elif gpu_memory_fraction == 0:
    #     sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
    # K.set_session(sess)


def get_optimizer(optimizer_type, learning_rate, lr_decay=0):
    optimizer_class = getattr(importlib.import_module("tensorflow.keras.optimizers"), optimizer_type)
    optimizer = optimizer_class(lr=learning_rate, decay=lr_decay)
    return optimizer


def normalize_hinge_output(x):
    return (x + 1) / 2


def get_loss_function(loss_function):
    if loss_function == 'FocalLoss':
        from focal_loss import FocalLoss
        focal_loss = FocalLoss()
        return focal_loss.compute_loss
    elif loss_function == 'HammingLoss':
        from hamming_loss import hamming_loss
        return hamming_loss
    else:
        loss_class = getattr(importlib.import_module("tensorflow.keras.losses"), loss_function)
        return loss_class()


def classify_image(img, model, multi_label_classification, target_size=(224, 224, 3)):
    # resize
    img = img / 255.
    img = resize(img, target_size)
    batch_x = np.expand_dims(img, axis=0)
    # normalize
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    batch_x = (batch_x - imagenet_mean) / imagenet_std
    # predict
    predictions = model.predict(batch_x)
    if multi_label_classification:
        predictions[predictions >= 0.5] = 1
        predictions[predictions < 0.5] = 0
    else:
        predictions = np.argmax(predictions, axis=1)
    return predictions


def get_evaluation_metrics(pred, labels, class_names, loss_function, threshold=0.5):
    if 'Hinge' in loss_function:
        pred = normalize_hinge_output(pred)
    current_auroc = []
    for i in range(len(class_names)):
        try:
            score = roc_auc_score(labels[:, i], pred[:, i])
        except ValueError:
            score = 0
        current_auroc.append(score)
        print(f"{i + 1}. {class_names[i]}: {score}")
    print("*********************************")

    mean_auroc = np.mean(current_auroc)
    print(f"mean auroc: {mean_auroc}")

    prec, rec, fscore, support = precision_recall_fscore_support(labels, pred >= threshold, average='macro')
    AP = average_precision_score(labels, pred)
    exact_accuracy = accuracy_score(labels, pred >= threshold)
    ham_loss = hamming_loss(labels, pred >= threshold)
    print(
        f"precision:{prec:.2f}, recall: {rec:.2f}, fscore: {fscore:.2f}, AP: {AP:.2f}, exact match accuracy: {exact_accuracy:.2f}, hamming loss: {ham_loss:.2f}")
    return mean_auroc, prec, rec, fscore, AP, exact_accuracy, ham_loss


def get_sample_counts(labels):
    total_count = labels.shape[0]
    positive_counts = np.sum(labels, axis=0)
    classes = []
    for i in range(labels.shape[1]):
        classes.append(str(i))
    class_positive_counts = dict(zip(classes, positive_counts))
    return total_count, class_positive_counts


def get_class_weights(labels, multiply):
    def get_single_class_weight(pos_counts, total_counts):
        denominator = (total_counts - pos_counts) * multiply + pos_counts
        return {
            0: pos_counts / denominator,
            1: (denominator - pos_counts) / denominator,
        }

    total_counts, class_positive_counts = get_sample_counts(labels)
    class_names = list(class_positive_counts.keys())
    label_counts = np.array(list(class_positive_counts.values()))
    class_weights = []
    for i, class_name in enumerate(class_names):
        class_weights.append(get_single_class_weight(label_counts[i], total_counts))

    return class_weights
