import logging
import os
import os.path
import warnings
from collections import namedtuple
from functools import lru_cache

import numpy as np
from nltk.metrics.distance import edit_distance

import finetune
from finetune.errors import FinetuneError
from finetune.base_models import RoBERTa

import tensorflow as tf

LOGGER = logging.getLogger("finetune")


def finetune_model_path(path):
    return os.path.abspath(
        os.path.join(os.path.dirname(finetune.__file__), "model", path)
    )


@lru_cache()
def all_gpus(visible_gpus=None):
    """
    Get integer ids of all available GPUs.
    """
    if visible_gpus is not None:
        return [int(gpu) for gpu in visible_gpus]
    devices = tf.config.experimental.list_physical_devices("GPU")
    LOGGER.info(
        " Visible GPUs: {{{}}}".format(
            ", ".join(
                [
                    "{}:{}".format(device.device_type, device.name).strip()
                    for device in devices
                ]
            )
        )
    )
    if not devices:
        warnings.warn("Failed to find available GPUS.  Falling back to CPU only mode.")
    device_ids = list(range(len(devices)))
    return device_ids


GridSearchable = namedtuple("GridSearchable", "default iterator")


class Settings(dict):
    """
    Model configuration options

    :param base_model: Which base model to use - one of {GPT, GPT2, RoBERTa, BERT, TextCNN, TCN}, imported from finetune.base_models. Defaults to `GPT`.
    :param batch_size: Number of examples per batch, defaults to `2`.
    :param visible_gpus: List of integer GPU ids to spread out computation across, defaults to all available GPUs.
    :param n_epochs: Number of iterations through training data, defaults to `3`.
    :param seed: Random seed to use for repeatability purposes, defaults to `42`.
    :param max_length:  Maximum number of subtokens per sequence. Examples longer than this number will be truncated
        (unless `chunk_long_sequences=True` for SequenceLabeler models). Defaults to `512`.
    :param weight_stddev: Standard deviation of initial weights.  Defaults to `0.02`.
    :param chunk_long_sequences: When True, use a sliding window approach to predict on
        examples that are longer than max length.  The progress bar will display the number of chunks processed rather than the number of examples. Defaults to `True`.  
    :param use_gpu_crf_predict: Use GPU op for crf predictions. Defaults to `auto`.
        examples that are longer than max length.  The progress bar will display the number of chunks processed rather than the number of examples. Defaults to `True`.
    :param chunk_context: How much context to include arround chunked text.
    :param chunk_alignment: Alignment of the active section of the chunks "left", "right", "center".
    :param low_memory_mode: When True, only store partial gradients on forward pass
        and recompute remaining gradients incrementally in order to save memory.  Defaults to `False`.
    :param float_16_predict: Whether to run prediction in float 16 mode, this is only available for bert based models and will likely only yield performance improvements on GPUs with native float16 support such as Volta and Tesla.
    :param optimize_for: Optimize auto parameters for either `accuracy`, `speed`, or `predict_speed` Defaults to `accuracy`
    :param embed_p_drop: Embedding dropout probability.  Defaults to `0.1`.
    :param attn_p_drop: Attention dropout probability.  Defaults to `0.1`.
    :param resid_p_drop: Residual layer fully connected network dropout probability.  Defaults to `0.1`.
    :param clf_p_drop: Classifier dropout probability.  Defaults to `0.1`.
    :param l2_reg: L2 regularization coefficient. Defaults to `0.01`.
    :param vector_l2: Whether to apply weight decay regularization to vectors (biases, normalization etc..). Defaults to False.
    :param optimizer: Optimizer to use, current options include AdamW or AdamaxW.
    :param b1: Adam b1 parameter.  Defaults to `0.9`.
    :param b2: Adam b2 parameter.  Defaults to `0.999`.
    :param epsilon: Adam epsilon parameter: Defaults to `1e-8`.
    :param lr_schedule: Learning rate schedule -- see `finetune/optimizers.py` for more options.
    :param lr: Learning rate.  Defaults to `6.25e-5`.
    :param lr_warmup: Learning rate warmup (percentage of all batches to warmup for).  Defaults to `0.002`.
    :param max_grad_norm: Clip gradients larger than this norm. Defaults to `1.0`.
    :param shuffle_buffer_size: How many examples to load into a buffer before shuffling. Defaults to `100`.
    :param dataset_size: Must be specified in order to calculate the learning rate schedule when the inputs provided are generators rather than static datasets.  
    :param accum_steps: Number of updates to accumulate before applying. This is used to simulate a higher batch size.
    :param lm_loss_coef: Language modeling loss coefficient -- a value between `0.0` - `1.0`
        that indicates how to trade off between language modeling loss
        and target model loss.  Usually not beneficial to turn on unless
        dataset size exceeds a few thousand examples.  Defaults to `0.0`.
    :param summarize_grads: Include gradient summary information in tensorboard.  Defaults to `False`.
    :param val_size: Validation set size if int. Validation set size as percentage of all training data if float.  Defaults to 0.  If value "auto" is provided, validation will not be run by default if n_examples < 50.
        If n_examples > 50, defaults to max(5, min(100, 0.05 * n_examples))
    :param val_interval: Evaluate on validation set after `val_interval` batches.
        Defaults to 4 * val_size / batch_size to ensure that too much time is not spent on validation.
    :param lm_temp: Language model temperature -- a value of `0.0` corresponds to greedy maximum likelihood predictions
        while a value of `1.0` corresponds to random predictions. Defaults to `0.6`.
    :param seq_num_heads: Number of attention heads of final attention layer. Defaults to `16`.
    :param keep_best_model: Whether or not to keep the highest-performing model weights throughout the train. Defaults to `False`.
    :param early_stopping_steps: How many steps to continue with no loss improvement before early stopping. Defaults to `None`.
    :param subtoken_predictions: Return predictions at subtoken granularity or token granularity?  Defaults to `False`.
    :param multi_label_sequences: Use a multi-labeling approach to sequence labeling to allow overlapping labels.
    :param multi_label_threshold: Threshold of sigmoid unit in multi label classifier.
        Can be increased or lowered to trade off precision / recall. Defaults to `0.5`.
    :param tensorboard_folder: Directory for tensorboard logs. Tensorboard logs will not be written
        unless tensorboard_folder is explicitly provided. Defaults to `None`.
    :param log_device_placement: Log which device each operation is placed on for debugging purposes.  Defaults to `False`.
    :param allow_soft_placement: Allow tf to allocate an operation to a different device if a device is unavailable.  Defaults to `True`.
    :param save_adam_vars: Save adam parameters when calling `model.save()`.  Defaults to `True`.
    :param num_layers_trained: How many layers to finetune.  Specifying a value less than model's number of layers will train layers starting from model output. Defaults to `12`.
    :param train_embeddings: Should embedding layer be finetuned? Defaults to `True`.
    :param class_weights: One of 'log', 'linear', or 'sqrt'. Auto-scales gradient updates based on class frequency.  Can also be a dictionary that maps from true class name to loss coefficient. Defaults to `None`.
    :param oversample: Should rare classes be oversampled?  Defaults to `False`.
    :param eval_acc: if True, calculates accuracy and writes it to the tensorboard summary files for valudation runs.
    :param save_dtype: specifies what precision to save model weights with.  Defaults to `np.float32`.
    :param regression_loss: the loss to use for regression models. One of `L1` or `L2`, defaults to `L2`.
    :param debugging_logs: if True, output tensorflow logs and turn off TQDM logging. Defaults to `False`.
    :param val_set: Where it is neccessary to use an explicit validation set, provide it here as a tuple (text, labels)
    :param per_process_gpu_memory_fraction: fraction of the overall amount of memory that each visible GPU should be allocated, defaults to `1.0`.
    :param max_empty_chunk_ratio: Controls the maximum ratio of empty to labeled chunks for sequence labeling. None includes all chunks, defaults to 1.0.

    """

    def get_grid_searchable(self):
        return self.grid_searchable

    def __init__(self, **kwargs):
        super().__init__()
        self.grid_searchable = {}
        for key, value in kwargs.items():
            self[key] = value

    def __getattr__(self, attr):
        if attr.startswith("__"):
            raise AttributeError

        if attr == "base_model_path":
            full_path = finetune_model_path(self["base_model_path"])
            if os.path.exists(full_path):
                return full_path

        return self[attr]

    def __setitem__(self, key, value):
        if isinstance(value, GridSearchable):
            self.grid_searchable[key] = value.iterator
            value = value.default
        return super().__setitem__(key, value)

    def __setattr__(self, k, v):
        return self.__setitem__(k, v)

    __delattr__ = dict.__delitem__


def did_you_mean(keyword, keyword_pool):
    candidates = list(keyword_pool)
    closest_match_idx = np.argmin(
        [edit_distance(keyword, candidate) for candidate in candidates]
    )
    return candidates[closest_match_idx]


def assert_valid_config(**kwargs):
    expected_keys = set(get_default_config().keys())
    for kwarg in kwargs:
        if kwarg not in expected_keys:
            raise FinetuneError(
                "Unexpected setting configuration: `{}` is an invalid keyword. "
                "Did you mean `{}`?".format(kwarg, did_you_mean(kwarg, expected_keys))
            )


def get_default_config():
    """
    Gets a config object containing all the default parameters for each variant of the model.

    :return: Config object.
    """
    # lazy import to avoid circular dependency
    from finetune import VERSION

    settings = Settings(
        # General Settings
        low_memory_mode=False,
        float_16_predict=False,
        save_adam_vars=False,
        shuffle_buffer_size=100,
        dataset_size=None,
        batch_size="auto",
        predict_batch_size="auto",
        visible_gpus=None,  # defaults to all available
        n_epochs="auto",
        seed=42,
        max_length="auto",
        weight_stddev=0.02,
        save_dtype=None,
        val_set=None,
        per_process_gpu_memory_fraction=None,
        distribution_strategy="central_storage",
        xla=False,
        optimize_for="accuracy",
        sort_by_length=True,
        collapse_whitespace=False,
        permit_uninitialized=None,
        #
        # Regularization
        embed_p_drop=0.1,
        attn_p_drop=0.1,
        resid_p_drop=0.1,
        clf_p_drop=0.1,
        l2_reg=GridSearchable(0.01, [0.0, 0.1, 0.01, 0.001]),
        vector_l2=False,
        #
        # Early Stopping and Validation
        keep_best_model=False,
        early_stopping_steps=None,
        eval_acc=False,
        val_size=0.0,
        val_interval=None,
        in_memory_finetune=None,
        #
        # Debugging
        log_device_placement=False,
        soft_device_placement=True,
        tensorboard_folder=None,
        summarize_grads=False,
        debugging_logs=False,
        cache_weights_to_file=False,
        #
        # Partial Fitting
        num_layers_trained=12,
        train_embeddings=True,
        #
        # Class Imbalance
        class_weights=None,
        oversample=False,
        #
        # Optimization Params
        optimizer="AdamW",
        b1=0.9,
        b2=0.999,
        epsilon=1e-8,
        lr_schedule="warmup_linear",
        lr=GridSearchable(6.25e-5, [6.25e-4, 6.25e-5, 6.25e-6]),
        lr_warmup=0.002,
        max_grad_norm=1.0,
        accum_steps=1,
        #
        # Language Model Settings
        lm_loss_coef=0.0,
        lm_temp=0.6,
        lm_type="lm",
        mask_proba=0.15,
        #
        # Masked Language Model Settings
        max_masked_tokens=128,
        #
        # Sequence Labeling
        seq_num_heads=16,
        pad_token="<PAD>",
        pad_idx=None,
        subtoken_predictions=False,
        multi_label_sequences=False,
        multi_label_threshold=0.5,
        chunk_long_sequences=True,
        chunk_context="auto",
        chunk_alignment="center",
        add_eos_bos_to_chunk=True,
        filter_empty_examples=False,
        crf_sequence_labeling=True,
        use_gpu_crf_predict="auto",
        max_empty_chunk_ratio=1.0,
        auto_negative_sampling=False,
        bio_tagging=False,
        #
        # Regression Params
        regression_loss="L2",
        #
        # Association Params
        viable_edges=None,
        association_types=None,
        assocation_loss_weight=100.0,
        #
        # Oscar only
        oscar_use_fp16=False,
        scale_loss=False,
        oscar_use_timing=True,
        oscar_feat_mode="final_state",
        oscar_use_fused_kernel=False,
        #
        # Location of model weights
        base_model=RoBERTa,
        base_model_path=None,
        #
        # Possible `SourceModel` specific settings
        n_heads=None,
        n_layer=None,
        act_fn=None,
        n_embed=None,
        #
        # for TCN SourceModel only
        n_filter=None,
        kernel_size=None,
        #
        # for TextCNN SourceModel only
        kernel_sizes=None,
        num_filters_per_size=None,
        n_embed_featurizer=None,  # needed because the dimensions CNN output are different from the embedding dimensions
        #
        # BERT only
        bert_intermediate_size=None,
        bert_use_pooler=True,
        bert_use_type_embed=True,
        #
        # Auxiliary Information
        use_auxiliary_info=False,
        default_context=None,
        context_dim=None,  # number of context dimensions to be inserted
        #
        # Document Representation
        context_injection=False,
        reading_order_removed=False,
        anneal_reading_order=False,
        context_channels=None,
        #
        # T5
        beam_size=1,
        beam_search_alpha=0.2,
        include_bos_eos=True,
        # Serialize finetune version with model
        version=VERSION,
    )
    return settings


def get_config(error_on_invalid_keywords=True, **kwargs):
    """
    Gets a config object containing all the default parameters for each variant of the model.

    :param **kwargs: Keyword arguments to override default values.
    :return: Config object.    """
    if error_on_invalid_keywords:
        assert_valid_config(**kwargs)
    config = get_default_config()
    config.base_model = kwargs.get("base_model", config.base_model)
    config.update(config.base_model.settings)
    config.update(kwargs)
    return config


def cpu_config(**kwargs):
    config = get_config(**kwargs)
    config.visible_gpus = []
    config.update(kwargs)
    return config
