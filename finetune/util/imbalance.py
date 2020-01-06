"""
Utilities for dealing with class imbalance
"""
import tensorflow as tf
import numpy as np
from collections import Counter

from finetune.encoding.target_encoders import LabelEncoder
from finetune.errors import FinetuneError


def compute_class_weights(class_weights, class_counts):
    """
    Optionally compute class weights based on target distribution
    """
    if class_weights is None:
        return

    options = {"linear", "log", "sqrt"}

    if isinstance(class_weights, str) and class_weights in options:
        counts = class_counts
        max_count = max(counts.values())
        computed_weights = {}
        for class_name, count in counts.items():
            ratio = max_count / count
            if class_weights == "linear":
                computed_weights[class_name] = ratio
            elif class_weights == "sqrt":
                computed_weights[class_name] = np.sqrt(ratio)
            elif class_weights == "log":
                computed_weights[class_name] = np.log(ratio) + 1
        class_weights = computed_weights

    if not isinstance(class_weights, dict):
        raise FinetuneError(
            "Invalid value for config.class_weights: {}. "
            "Expected dictionary mapping from class name to weight or one of {}".format(
                class_weights, list(options)
            )
        )
    return class_weights


def class_weight_tensor(class_weights, target_dim, label_encoder):
    """
    Convert from dictionary of class weights to tf tensor
    """
    class_weight_arr = np.ones(target_dim, dtype=np.float32)
    for i, cls in enumerate(label_encoder.target_labels):
        class_weight_arr[i] = class_weights.get(cls, 1.0)

    class_weight_tensor = tf.convert_to_tensor(class_weight_arr)
    return class_weight_tensor


def class_count_tensor(class_counts, target_dim, label_encoder):
    """
    Convert from dictionary of class counts to tf tensor
    """
    class_count_arr = np.ones(target_dim, dtype=np.float32)
    for i, cls in enumerate(label_encoder.target_labels):
        class_count_arr[i] = class_counts.get(cls, 1.0)
    
    class_count_tensor = tf.convert_to_tensor(class_count_arr)
    return class_count_tensor