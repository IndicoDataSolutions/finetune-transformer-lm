import functools
import tensorflow as tf
from tensorflow_addons.text.crf import crf_log_likelihood

from finetune.base_models.gpt.featurizer import attn, dropout, norm
from finetune.util.shapes import shape_list, merge_leading_dims
from finetune.optimizers.recompute_grads import recompute_grad
from finetune.errors import FinetuneError
from finetune.nn.activations import act_fns
from finetune.nn.nn_utils import norm
from tensorflow.python.framework import function


def perceptron(x, ny, config, w_init=None, b_init=None):
    """
    A very standard linear Perceptron model.
    :param x: Input tensor.
    :param ny: Number of outputs.
    :param config: A config object.
    :param w_init: Weight initializer.
    :param b_init: Bias initializer.
    :return: The output of the perceptron model.
    """
    w_init = w_init or tf.compat.v1.random_normal_initializer(
        stddev=config.weight_stddev
    )
    b_init = b_init or tf.compat.v1.constant_initializer(0)

    with tf.compat.v1.variable_scope("perceptron"):
        nx = config.n_embed
        w = tf.compat.v1.get_variable("w", [nx, ny], initializer=w_init)
        b = tf.compat.v1.get_variable("b", [ny], initializer=b_init)
        return tf.matmul(x, w) + b


def masked_language_model(
    *,
    X,
    mlm_weights,
    mlm_ids,
    mlm_positions,
    embed_weights,
    hidden,
    config,
    reuse=None,
    train=False
):

    with tf.compat.v1.variable_scope("model/masked-language-model"):
        batch, seq, feats = shape_list(hidden)
        flat_offsets = tf.reshape(tf.range(0, batch, dtype=tf.int32) * seq, [-1, 1])

        not_padding = tf.reshape(mlm_weights, [-1]) > 1e-9
        flat_positions = tf.boolean_mask(
            tensor=tf.reshape(mlm_positions + flat_offsets, [-1]), mask=not_padding
        )  # take off the padding entirely
        gathered_hidden = tf.gather(
            tf.reshape(hidden, [batch * seq, feats]), flat_positions
        )
        mlm_ids = tf.boolean_mask(tensor=tf.reshape(mlm_ids, [-1]), mask=not_padding)

        final_proj_w = tf.compat.v1.get_variable(
            "dense/kernel",
            [config.n_embed, config.n_embed],
            initializer=tf.compat.v1.random_normal_initializer(
                stddev=config.weight_stddev
            ),
        )
        final_proj_b = tf.compat.v1.get_variable(
            "dense/bias", [config.n_embed], initializer=tf.compat.v1.zeros_initializer
        )
        final_proj = act_fns[config.act_fn](
            tf.matmul(gathered_hidden, final_proj_w, transpose_b=True) + final_proj_b
        )

        normed_proj = norm(final_proj, "LayerNorm")
        n_vocab = shape_list(embed_weights)[0]
        output_bias = tf.compat.v1.get_variable(
            "output_bias", shape=[n_vocab], initializer=tf.compat.v1.zeros_initializer()
        )

        logits = tf.matmul(normed_proj, embed_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)

        mlm_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=mlm_ids,
        )  # No weights needed as there is no padding.

        logits = tf.scatter_nd(
            indices=flat_positions, updates=logits, shape=[batch * seq, n_vocab]
        )

        return {
            "logits": logits,
            "losses": mlm_loss,
        }


def language_model(
    *, X, sequence_lengths, embed_weights, hidden, config, reuse=None, train=False
):
    """
    A language model output and loss for the language modelling objective described in the original finetune paper.
    This language model uses weights that are tied to the input embedding.
    :param X: The raw token ids fed to the featurizer.
    :param M: A loss mask, with 1's where losses should be counted and 0's elsewhere.
    :param embed_weights: The word embedding matrix, normally the one returned by the featurizer.
    :param hidden: Output of the featurizer.
    :param config: A config object.
    :param reuse: A Flag passed through to the tf.variable_scope context manager.
    :return: A dict containing:
        logits: The un-normalised log-probabilities over each word in the vocabulary.
        loss: The masked language modelling loss.

    """
    X = merge_leading_dims(X, 2)
    M = tf.sequence_mask(sequence_lengths, dtype=tf.float32)
    hidden = merge_leading_dims(hidden, 3)

    batch, seq = shape_list(X)
    vocab_size, hidden_dim = shape_list(embed_weights)

    with tf.compat.v1.variable_scope("model/language-model", reuse=reuse):
        # language model ignores last hidden state because we don't have a target
        lm_h = tf.reshape(
            hidden, [-1, config.n_embed]
        )  # [batch, seq_len, embed] --> [batch * seq_len, embed]
        lm_logits = tf.matmul(lm_h, embed_weights, transpose_b=True)  # tied weights
        lm_logits = tf.cast(lm_logits, tf.float32)
        hidden_shape = tf.shape(input=hidden)
        logits = tf.reshape(
            lm_logits, shape=tf.concat([hidden_shape[:-1], [vocab_size]], axis=0)
        )
        lm_logits_offset = tf.reshape(logits[:, :-1], [-1, vocab_size])

        lm_losses = tf.compat.v1.losses.sparse_softmax_cross_entropy(
            logits=lm_logits_offset,
            labels=tf.reshape(X[:, 1:], [-1]),
            weights=tf.reshape(M[:, 1:], [-1]),
        )

        perplexity = tf.math.divide_no_nan(
            tf.reduce_sum(input_tensor=tf.exp(lm_losses) * M[:, 1:], axis=1),
            tf.reduce_sum(input_tensor=M[:, 1:], axis=1),
        )

        return {
            "logits": logits,
            "losses": lm_losses,
            "perplexity": perplexity,
        }


def _apply_class_weight(losses, targets, class_weights=None):
    if class_weights is not None:
        # loss multiplier applied based on true class
        weights = tf.reduce_sum(
            input_tensor=class_weights * tf.cast(targets, dtype=tf.float32), axis=1
        )
        weights *= tf.math.divide_no_nan(
            tf.cast(
                tf.reduce_prod(input_tensor=tf.shape(input=weights)), dtype=tf.float32
            ),
            tf.reduce_sum(input_tensor=weights),
        )
        losses *= tf.expand_dims(weights, 1)
    return losses


def _apply_multilabel_class_weight(losses, targets, class_weights=None):
    if class_weights is not None:
        # loss multiplier applied based on true class
        weights = (
            # contribution of positive class
            class_weights * tf.cast(targets, dtype=tf.float32)
            +
            # contribution of negative class
            tf.ones_like(class_weights) * (1 - tf.cast(targets, dtype=tf.float32))
        )
        weights *= tf.math.divide_no_nan(
            tf.cast(
                tf.reduce_prod(input_tensor=tf.shape(input=weights)), dtype=tf.float32
            ),
            tf.reduce_sum(input_tensor=weights),
        )
        losses *= weights
    return losses


def classifier(hidden, targets, n_targets, config, train=False, reuse=None, **kwargs):
    """
    A simple linear classifier.

    :param hidden: The output of the featurizer. [batch_size, embed_dim]
    :param targets: One hot encoded target ids. [batch_size, n_classes]
    :param n_targets: A python int containing the number of classes that the model should be learning to predict over.
    :param dropout_placeholder:
    :param config: A config object, containing all parameters for the featurizer.
    :param train: If this flag is true, dropout and losses are added to the graph.
    :param reuse: Should reuse be set within this scope.
    :param kwargs: Spare arguments.
    :return: dict containing:
        logits: The unnormalised log probabilities of each class.
        losses: The loss for the classifier.
    """
    with tf.compat.v1.variable_scope("classifier", reuse=reuse):
        hidden = dropout(hidden, config.clf_p_drop, train)
        clf_logits = perceptron(hidden, n_targets, config)
        if targets is None:
            clf_losses = None
        else:
            clf_losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=clf_logits, labels=tf.stop_gradient(targets)
            )

            clf_losses = _apply_class_weight(
                clf_losses, targets, kwargs.get("class_weights")
            )

        return {"logits": clf_logits, "losses": clf_losses}


def multi_choice_question(
    hidden, targets, n_targets, config, train=False, reuse=None, **kwargs
):
    with tf.compat.v1.variable_scope("model", reuse=reuse):
        if targets is not None:
            targets = tf.cast(targets, tf.int32)
        hidden = dropout(hidden, config.clf_p_drop, train)
        hidden = tf.unstack(hidden, num=n_targets, axis=1)
        hidden = tf.concat(hidden, axis=0)

        clf_out = perceptron(hidden, 1, config)
        clf_out = tf.split(clf_out, n_targets, axis=0)
        clf_out = tf.concat(clf_out, 1)

        if targets is None:
            clf_losses = None
        else:
            clf_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=clf_out, labels=tf.stop_gradient(targets)
            )

            clf_losses = _apply_class_weight(
                clf_losses, targets, kwargs.get("class_weights")
            )

        return {"logits": clf_out, "losses": clf_losses}


def multi_classifier(
    hidden, targets, n_targets, config, train=False, reuse=None, **kwargs
):
    """
    A simple linear classifier.

    :param hidden: The output of the featurizer. [batch_size, embed_dim]
    :param targets: The placeholder representing the sparse targets [batch_size, n_targets]
    :param n_targets: A python int containing the number of classes that the model should be learning to predict over.
    :param config: A config object, containing all parameters for the featurizer.
    :param train: If this flag is true, dropout and losses are added to the graph.
    :param reuse: Should reuse be set within this scope.
    :param kwargs: Spare arguments.
    :return: dict containing:
        logits: The unnormalised log probabilities of each class.
        losses: The loss for the classifier.
    """
    with tf.compat.v1.variable_scope("model", reuse=reuse):
        hidden = dropout(hidden, config.clf_p_drop, train)
        clf_logits = perceptron(hidden, n_targets, config)
        if targets is None:
            clf_losses = None
        else:
            clf_losses = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=clf_logits, labels=tf.stop_gradient(targets)
            )
            clf_losses = _apply_multilabel_class_weight(
                clf_losses, targets, kwargs.get("class_weights")
            )
        return {"logits": clf_logits, "losses": clf_losses}


def regressor(hidden, targets, n_targets, config, train=False, reuse=None, **kwargs):
    """
    A simple linear regressor.

    :param hidden: The output of the featurizer. [batch_size, embed_dim]
    :param targets: The placeholder representing the regression targets. [batch_size]
    :param n_targets: A python int containing the number of outputs that the model should be learning to predict over.
    :param dropout_placeholder:
    :param config: A config object, containing all parameters for the featurizer.
    :param train: If this flag is true, dropout and losses are added to the graph.
    :param reuse: Should reuse be set within this scope.
    :param kwargs: Spare arguments.
    :return: dict containing:
        logits: The regression outputs.
        losses: L2 Loss for the regression targets.
    """
    with tf.compat.v1.variable_scope("regressor", reuse=reuse):
        hidden = dropout(hidden, config.clf_p_drop, train)
        outputs = perceptron(hidden, n_targets, config)
        if targets is None:
            loss = None
        else:
            if config.regression_loss.upper() == "L2":
                loss = tf.nn.l2_loss(outputs - targets)
            elif config.regression_loss.upper() == "L1":
                loss = tf.abs(outputs - targets)
            else:
                raise FinetuneError(
                    "regression_loss needs to be either L1 or L2, instead it is {}".format(
                        config.regression_loss
                    )
                )
        return {"logits": outputs, "losses": loss}


def ordinal_regressor(
    hidden,
    targets,
    n_targets,
    config,
    shared_threshold_weights=True,
    train=False,
    reuse=None,
    **kwargs
):
    """
    Ordinal Regressor using all-threshold loss.

    :param hidden: The output of the featurizer. [batch_size, embed_dim]
    :param targets: The placeholder representing the regression targets (binary threshold values). [batch_size]
    :param n_targets: A python int containing the number of thresholds that the model should be learning to predict over.
    :param dropout_placeholder:
    :param config: A config object, containing all parameters for the featurizer.
    :param train: If this flag is true, dropout and losses are added to the graph.
    :param reuse: Should reuse be set within this scope.
    :param kwargs: Spare arguments.
    :return: dict containing:
        logits: The regression outputs.
        losses: All-threshold Loss for the regression targets.
    """
    with tf.compat.v1.variable_scope("ordinalregressor", reuse=reuse):
        hidden = dropout(hidden, config.clf_p_drop, train)
        if shared_threshold_weights:
            w_init = tf.compat.v1.random_normal_initializer(stddev=config.weight_stddev)
            b_init = tf.compat.v1.random_normal_initializer(0)
            nx = config.n_embed
            w = tf.compat.v1.get_variable("w", [nx, 1], initializer=w_init)
            b = tf.compat.v1.get_variable("b", [n_targets], initializer=b_init)
            logits = tf.matmul(hidden, w) + b
        else:
            logits = perceptron(hidden, n_targets, config)

        if targets is None:
            outputs = tf.sigmoid(logits)
            loss = None
        else:
            outputs = logits
            loss = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits, labels=tf.stop_gradient(targets)
            )
        return {"logits": outputs, "losses": loss}


def class_reweighted_grad(logits, class_weights):
    def custom_grad_fn(op, g):
        new_g = g * class_weights
        ratio = tf.math.divide_no_nan(tf.norm(g), tf.norm(new_g))
        return [new_g * ratio]

    @function.Defun(
        logits.dtype,
        func_name="class_reweight_grad",
        python_grad_func=custom_grad_fn,
        shape_func=lambda _: [logits.get_shape()],
    )
    def identity(l):
        return tf.identity(l)

    return identity(logits)


def sequence_labeler(
    hidden,
    targets,
    n_targets,
    config,
    pad_id,
    multilabel=False,
    train=False,
    reuse=None,
    lengths=None,
    use_crf=True,
    **kwargs
):
    """
    An Attention based sequence labeler model.

    In the case of unidirectional base models such as GPT this model takes the output of the pre-trained model,
    applies an additional randomly initialised multihead attention block, with residuals on top.
    The extra attention is not future masked to allow the model to label sequences based on context in both directions.
    The representations fed into this model are necessarily future masked because a language modelling loss is the
    original objective of the featurizer.

    For bidirectional base models we apply the crf model directly to the output of the base model.

    :param hidden: The output of the featurizer. [batch_size, sequence_length, embed_dim]
    :param targets: The placeholder representing the sequence labeling targets. [batch_size, sequence_length]
    :param n_targets: A python int containing the number of classes that the model should be learning to predict over.
    :param config: A config object, containing all parameters for the featurizer.
    :param train: If this flag is true, dropout and losses are added to the graph.
    :param reuse: Should reuse be set within this scope.
    :param lengths: The number of non-padding tokens in the input.
    :param kwargs: Spare arguments.
    :return: dict containing:
        "logits": The un-normalised log probabilities of each class being in each location. For usable predictions,
            sampling from this distribution is not sufficient and a viterbi decoding method should be used.
        "losses": The negative log likelihood for the sequence targets.
        "predict_params": A dictionary of params to be fed to the viterbi decode function.
    """
    with tf.compat.v1.variable_scope("sequence-labeler", reuse=reuse):

        if targets is not None:
            targets = tf.cast(targets, dtype=tf.int32)

        nx = config.n_embed

        def seq_lab_internal(hidden):
            if config.base_model.is_bidirectional:
                n = hidden
            else:
                attn_fn = functools.partial(
                    attn,
                    scope="seq_label_attn",
                    n_state=nx,
                    n_head=config.seq_num_heads,
                    resid_pdrop=config.resid_p_drop,
                    attn_pdrop=config.attn_p_drop,
                    train=train,
                    scale=False,
                    mask=False,
                )
                n = norm(attn_fn(hidden) + hidden, "seq_label_residual")

            flat_logits = tf.compat.v1.layers.dense(n, n_targets)
            logits = tf.reshape(
                flat_logits, tf.concat([tf.shape(input=hidden)[:2], [n_targets]], 0)
            )
            return logits

        with tf.compat.v1.variable_scope("seq_lab_attn"):
            if config.low_memory_mode and train:
                seq_lab_internal = recompute_grad(
                    seq_lab_internal, use_entire_scope=True
                )
            logits = seq_lab_internal(hidden)
            logits = tf.cast(logits, tf.float32)  # always run the crf in float32

        loss = 0.0

        default_lengths = tf.shape(input=hidden)[1] * tf.ones(
            tf.shape(input=hidden)[0], dtype=tf.int32
        )
        if lengths is None:
            lengths = default_lengths

        class_weights = kwargs.get("class_weights")

        with tf.device("CPU:0" if train else logits.device):
            if multilabel:
                transition_params = []
                logits_individual = tf.unstack(logits, n_targets, axis=-1)
                if targets is not None:
                    targets_individual = tf.unstack(targets, n_targets, axis=-1)
                logits = []
                for i in range(n_targets):
                    transition_params.append(
                        tf.cast(
                            tf.compat.v1.get_variable(
                                "Transition_matrix_{}".format(i), shape=[2, 2]
                            ),
                            tf.float32,
                        )
                    )
                    logits.append(
                        tf.stack(
                            (logits_individual[pad_id], logits_individual[i]), axis=-1
                        )
                    )
                    if targets is not None and i != pad_id:
                        if class_weights is not None:
                            is_pos_cls = tf.cast(
                                targets_individual[i], dtype=tf.float32
                            )
                            class_weight = tf.expand_dims(
                                class_weights[i] * is_pos_cls
                                + class_weights[pad_id] * (1.0 - is_pos_cls),
                                -1,
                            )
                            logits_i = class_reweighted_grad(logits[-1], class_weight)
                        else:
                            logits_i = logits[i]
                        if use_crf:
                            loss -= crf_log_likelihood(
                                logits_i,
                                targets_individual[i],
                                lengths,
                                transition_params=transition_params[-1],
                            )[0]
                        else:
                            weights = tf.math.divide_no_nan(
                                tf.sequence_mask(
                                    lengths,
                                    maxlen=tf.shape(input=targets_individual[i])[1],
                                    dtype=tf.float32,
                                ),
                                tf.expand_dims(tf.cast(lengths, tf.float32), -1),
                            )
                            loss += tf.compat.v1.losses.sparse_softmax_cross_entropy(
                                targets_individual[i], logits_i, weights=weights
                            )
                logits = tf.stack(logits, axis=-1)
            else:
                if class_weights is not None and train:
                    class_weights = tf.reshape(class_weights, [1, 1, -1])
                    one_hot_class_weights = class_weights * tf.one_hot(
                        targets, depth=n_targets
                    )
                    per_token_weights = tf.reduce_sum(
                        input_tensor=one_hot_class_weights, axis=-1, keepdims=True
                    )
                    logits = class_reweighted_grad(logits, per_token_weights)

                transition_params = tf.cast(
                    tf.compat.v1.get_variable(
                        "Transition_matrix", shape=[n_targets, n_targets]
                    ),
                    tf.float32,
                )
                if targets is not None:
                    if use_crf:
                        log_likelihood, _ = crf_log_likelihood(
                            logits,
                            targets,
                            lengths,
                            transition_params=transition_params,
                        )
                        loss = -log_likelihood
                    else:
                        weights = tf.math.divide_no_nan(
                            tf.sequence_mask(
                                lengths,
                                maxlen=tf.shape(input=targets)[1],
                                dtype=tf.float32,
                            ),
                            tf.expand_dims(tf.cast(lengths, tf.float32), -1),
                        )
                        loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(
                            targets, logits, weights=weights
                        )

        return {
            "logits": logits,
            "losses": loss,
            "predict_params": {
                "transition_matrix": transition_params,
                "sequence_length": lengths,
            },
        }

def multi_crf_group_labeler(
    hidden,
    targets,
    n_targets,
    config,
    pad_id,
    train=False,
    reuse=None,
    lengths=None,
    use_crf=False,
    **kwargs
):
    """
    Multi CRF group tagging model. Takes two sets of targets - one for normal
    tagging and one for group tagging. Learns a CRF for each set of targets

    :param hidden: The output of the featurizer. [batch_size, sequence_length, embed_dim]
    :param targets: The placeholder representing the sequence labeling targets. [batch_size, 2, sequence_length]
    :param n_targets: A python int containing the number of classes that the model should be learning to predict over.
    :param config: A config object, containing all parameters for the featurizer.
    :param train: If this flag is true, dropout and losses are added to the graph.
    :param reuse: Should reuse be set within this scope.
    :param lengths: The number of non-padding tokens in the input.
    :param kwargs: Spare arguments.
    :return: dict containing:
        "logits": The un-normalised log probabilities of each class being in each location. For usable predictions,
            sampling from this distribution is not sufficient and a viterbi decoding method should be used.
        "losses": The negative log likelihood for the sequence targets.
        "predict_params": A dictionary of params to be fed to the viterbi decode function.
    """
    with tf.compat.v1.variable_scope("multi-crf-group", reuse=reuse):

        if targets is not None:
            targets = tf.cast(targets, dtype=tf.int32)

        nx = config.n_embed

        def seq_lab_internal(hidden):
            flat_logits = tf.compat.v1.layers.dense(hidden, n_targets)
            logits = tf.reshape(
                flat_logits, tf.concat([tf.shape(input=hidden)[:2], [n_targets]], 0)
            )
            return logits

        def group_seq_lab_internal(hidden):
            flat_logits = tf.compat.v1.layers.dense(hidden, 3)
            logits = tf.reshape(
                flat_logits, tf.concat([tf.shape(input=hidden)[:2], [3]], 0)
            )
            return logits

        with tf.compat.v1.variable_scope("seq_lab_attn"):
            if config.low_memory_mode and train:
                seq_lab_internal = recompute_grad(
                    seq_lab_internal, use_entire_scope=True
                )
            logits = seq_lab_internal(hidden)
            logits = tf.cast(logits, tf.float32)  # always run the crf in float32
        with tf.compat.v1.variable_scope("group_seq_lab_attn"):
            if config.low_memory_mode and train:
                group_seq_lab_internal = recompute_grad(
                    group_seq_lab_internal, use_entire_scope=True
                )
            group_logits = group_seq_lab_internal(hidden)
            group_logits = tf.cast(group_logits, tf.float32)

        loss = 0.0

        default_lengths = tf.shape(input=hidden)[1] * tf.ones(
            tf.shape(input=hidden)[0], dtype=tf.int32
        )
        if lengths is None:
            lengths = default_lengths

        class_weights = kwargs.get("class_weights")

        with tf.device("CPU:0" if train else logits.device):
            if class_weights is not None and train:
                class_weights = tf.reshape(class_weights, [1, 1, -1])
                one_hot_class_weights = class_weights * tf.one_hot(
                    targets[:, 0, :], depth=n_targets
                )
                per_token_weights = tf.reduce_sum(
                    input_tensor=one_hot_class_weights, axis=-1, keepdims=True
                )
                logits = class_reweighted_grad(logits, per_token_weights)

            transition_params = tf.cast(
                tf.compat.v1.get_variable(
                    "Transition_matrix", shape=[n_targets, n_targets]
                ),
                tf.float32,
            )
            group_transition_params = tf.cast(
                tf.compat.v1.get_variable(
                    "Group_transition_matrix", shape=[3, 3]
                ),
                tf.float32,
            )
            if targets is not None:
                if use_crf:
                    log_likelihood, _ = crf_log_likelihood(
                        logits,
                        targets[:, 0, :],
                        lengths,
                        transition_params=transition_params,
                    )
                    group_log_likelihood, _ = crf_log_likelihood(
                        group_logits,
                        targets[:, 1, :],
                        lengths,
                        transition_params=group_transition_params,
                    )
                    loss = -(log_likelihood + group_log_likelihood)
                else:
                    weights = tf.math.divide_no_nan(
                        tf.sequence_mask(
                            lengths,
                            maxlen=tf.shape(input=targets)[2],
                            dtype=tf.float32,
                        ),
                        tf.expand_dims(tf.cast(lengths, tf.float32), -1),
                    )
                    loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(
                        targets[:, 2, :], logits, weights=weights
                    )
                    group_loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(
                        targets[:, 1, :], group_logits, weights=weights
                    )
                    loss += group_loss

        return {
            "logits": [logits, group_logits],
            "losses": loss,
            "predict_params": {
                "transition_matrix": transition_params,
                "group_transition_matrix": group_transition_params,
                "sequence_length": lengths,
            },
        }

def multi_logit_group_labeler (
    hidden,
    targets,
    n_targets,
    config,
    pad_id,
    train=False,
    reuse=None,
    lengths=None,
    use_crf=False,
    **kwargs
):
    """
    Multi CRF group tagging model. Takes two sets of targets - one for normal
    tagging and one for group tagging. Learns a CRF for each set of targets

    :param hidden: The output of the featurizer. [batch_size, sequence_length, embed_dim]
    :param targets: The placeholder representing the sequence labeling targets. [batch_size, 2, sequence_length]
    :param n_targets: A python int containing the number of classes that the model should be learning to predict over.
    :param config: A config object, containing all parameters for the featurizer.
    :param train: If this flag is true, dropout and losses are added to the graph.
    :param reuse: Should reuse be set within this scope.
    :param lengths: The number of non-padding tokens in the input.
    :param kwargs: Spare arguments.
    :return: dict containing:
        "logits": The un-normalised log probabilities of each class being in each location. For usable predictions,
            sampling from this distribution is not sufficient and a viterbi decoding method should be used.
        "losses": The negative log likelihood for the sequence targets.
        "predict_params": A dictionary of params to be fed to the viterbi decode function.
    """
    assert n_targets % 3 == 0, f"{n_targets} classes not divisible by 3!"
    with tf.compat.v1.variable_scope("multi-logit-group", reuse=reuse):

        if targets is not None:
            targets = tf.cast(targets, dtype=tf.int32)

        nx = config.n_embed

        def seq_lab_internal(hidden):
            flat_logits = tf.compat.v1.layers.dense(hidden, n_targets // 3)
            logits = tf.reshape(
                flat_logits, tf.concat([tf.shape(input=hidden)[:2], [n_targets // 3]], 0)
            )
            return logits

        def group_seq_lab_internal(hidden):
            flat_logits = tf.compat.v1.layers.dense(hidden, 3)
            logits = tf.reshape(
                flat_logits, tf.concat([tf.shape(input=hidden)[:2], [3]], 0)
            )
            return logits

        with tf.compat.v1.variable_scope("seq_lab_attn"):
            if config.low_memory_mode and train:
                seq_lab_internal = recompute_grad(
                    seq_lab_internal, use_entire_scope=True
                )
            ner_logits = seq_lab_internal(hidden)
            ner_logits = tf.cast(ner_logits, tf.float32)  # always run the crf in float32
        with tf.compat.v1.variable_scope("group_seq_lab_attn"):
            if config.low_memory_mode and train:
                group_seq_lab_internal = recompute_grad(
                    group_seq_lab_internal, use_entire_scope=True
                )
            group_logits = group_seq_lab_internal(hidden)
            group_logits = tf.cast(group_logits, tf.float32)

        # Broadcast probabilities to make [batch, seq_len, n_classes] matrix
        final_shape = tf.concat((tf.shape(hidden)[:2], [n_targets]), 0)
        # [batch, seq_len, n_classes / 3, 3]
        logits = tf.expand_dims(ner_logits, 3) * tf.expand_dims(group_logits, 2)
        # [batch, seq_len, n_classes]
        logits = tf.reshape(logits, final_shape)
        # Note, in order for loss to work correctly the targets must be in the
        # form [AA-TAG1, BB-TAG1, CC-TAG1, AA-TAG2, BB-TAG2, CC-TAG2, AA-TAG3 ...]
        # where tags are grouped together and always in the same order of
        # prefixes, as this is the form the logits will be produced in via broadcasting

        loss = 0.0

        default_lengths = tf.shape(input=hidden)[1] * tf.ones(
            tf.shape(input=hidden)[0], dtype=tf.int32
        )
        if lengths is None:
            lengths = default_lengths

        class_weights = kwargs.get("class_weights")

        with tf.device("CPU:0" if train else logits.device):
            if class_weights is not None and train:
                class_weights = tf.reshape(class_weights, [1, 1, -1])
                one_hot_class_weights = class_weights * tf.one_hot(
                    targets, depth=n_targets
                )
                per_token_weights = tf.reduce_sum(
                    input_tensor=one_hot_class_weights, axis=-1, keepdims=True
                )
                logits = class_reweighted_grad(logits, per_token_weights)

            transition_params = tf.cast(
                tf.compat.v1.get_variable(
                    "Transition_matrix", shape=[n_targets, n_targets]
                ),
                tf.float32,
            )
            if targets is not None:
                if use_crf:
                    log_likelihood, _ = crf_log_likelihood(
                        logits,
                        targets,
                        lengths,
                        transition_params=transition_params,
                    )
                    loss = -log_likelihood
                else:
                    weights = tf.math.divide_no_nan(
                        tf.sequence_mask(
                            lengths,
                            maxlen=tf.shape(input=targets)[1],
                            dtype=tf.float32,
                        ),
                        tf.expand_dims(tf.cast(lengths, tf.float32), -1),
                    )
                    loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(
                        targets, logits, weights=weights
                    )
                    loss += group_loss

        return {
            "logits": logits,
            "losses": loss,
            "predict_params": {
                "transition_matrix": transition_params,
                "sequence_length": lengths,
            },
        }

def bros_decoder(
    hidden,
    targets,
    n_targets,
    config,
    pad_id,
    train=False,
    reuse=None,
    lengths=None,
    **kwargs
):
    """
    A BROS decoder.

    :param hidden: The output of the featurizer. [batch_size, sequence_length, embed_dim]
    :param targets: The placeholder representing the targets.  [batch_size, 2, sequence_length]
    :param n_targets: A python int containing the number of classes that the model should be learning to predict over.
    :param config: A config object, containing all parameters for the featurizer.
    :param train: If this flag is true, dropout and losses are added to the graph.
    :param reuse: Should reuse be set within this scope.
    :param lengths: The number of non-padding tokens in the input.
    :param kwargs: Spare arguments.
    :return: dict containing:
        "logits": The un-normalised log probabilities of each class being in each location. For usable predictions,
            sampling from this distribution is not sufficient and a viterbi decoding method should be used.
        "losses": The negative log likelihood for the sequence targets.
        "predict_params": A dictionary of params to be fed to the viterbi decode function.
    """
    with tf.compat.v1.variable_scope("bros-decoder", reuse=reuse):

        if targets is not None:
            targets = tf.cast(targets, dtype=tf.int32)

        nx = config.n_embed
        hidden_size = config.relation_hidden_size

        def get_out_logits(hidden):
            flat_logits = tf.compat.v1.layers.dense(hidden, 2)
            logits_shape = tf.concat([tf.shape(hidden)[:2], [2]], 0)
            logits = tf.reshape(flat_logits, logits_shape)
            return logits
        def get_out_hidden(hidden):
            flat_logits = tf.compat.v1.layers.dense(hidden, hidden_size)
            logits_shape = tf.concat([tf.shape(hidden)[:2], [hidden_size]], 0)
            logits = tf.reshape(flat_logits, logits_shape)
            return logits

        with tf.compat.v1.variable_scope("start_token_logits"):
            # [Batch Size, Sequence Length, 2]
            if config.low_memory_mode and train:
                get_out_logits = recompute_grad(get_out_logits, use_entire_scope=True)
            start_token_logits = get_out_logits(hidden)
            start_token_logits = tf.cast(start_token_logits, tf.float32)
        with tf.compat.v1.variable_scope("start_token_hidden"):
            if config.low_memory_mode and train:
                get_out_hidden = recompute_grad(get_out_hidden, use_entire_scope=True)
            # [Batch Size, Sequence Length, Hidden Size]
            start_token_hidden = get_out_hidden(hidden)
            start_token_hidden = tf.cast(start_token_hidden, tf.float32)
        with tf.compat.v1.variable_scope("next_token_hidden"):
            if config.low_memory_mode and train:
                get_out_hidden = recompute_grad(get_out_hidden, use_entire_scope=True)
            # [Batch Size, Sequence Length, Hidden Size]
            next_token_hidden = get_out_hidden(hidden)
            next_token_hidden = tf.cast(next_token_hidden, tf.float32)

            no_next_hidden = tf.cast(
                tf.compat.v1.get_variable(
                    "no_next_hidden", shape=[hidden_size]
                ),
                tf.float32,
            )
            no_next_hidden = tf.reshape(no_next_hidden, (1, 1, hidden_size))
            no_next_hidden = tf.repeat(no_next_hidden, tf.shape(hidden)[0], axis=0)
            # [Batch Size, Sequence Length + 1, Hidden Size]
            # Note: The no relation embedding comes first
            next_token_hidden = tf.concat((no_next_hidden, next_token_hidden), axis=1)
        with tf.compat.v1.variable_scope("next_token_logits"):
            # [Batch Size, Sequence Length, Sequence Legth + 1]
            next_token_logits = tf.matmul(
                start_token_hidden,
                next_token_hidden,
                transpose_b=True
            )

        if lengths is None:
            lengths = tf.shape(input=hidden)[1] * tf.ones(
                tf.shape(input=hidden)[0], dtype=tf.int32
            )

        loss = 0.0
        # class_weights = kwargs.get("class_weights")
        with tf.device("CPU:0" if train else start_token_logits.device):
            # if class_weights is not None and train:
            #     class_weights = tf.reshape(class_weights, [1, 1, -1])
            #     one_hot_class_weights = class_weights * tf.one_hot(
            #         targets, depth=n_targets
            #     )
            #     per_token_weights = tf.reduce_sum(
            #         input_tensor=one_hot_class_weights, axis=-1, keepdims=True
            #     )
            #     logits = class_reweighted_grad(logits, per_token_weights)

            if targets is not None:
                weights = tf.math.divide_no_nan(
                    tf.sequence_mask(
                        lengths,
                        maxlen=tf.shape(input=targets)[2],
                        dtype=tf.float32,
                    ),
                    tf.expand_dims(tf.cast(lengths, tf.float32), -1),
                )
                start_token_loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(
                    targets[:, 0, :], start_token_logits, weights=weights
                )
                next_token_loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(
                    targets[:, 1, :], next_token_logits, weights=weights
                )

                loss = start_token_loss + next_token_loss

        return {
            "logits": {
                "start_token_logits": start_token_logits,
                "next_token_logits": next_token_logits,
            },
            "losses": loss,
            "predict_params": {
                "sequence_length": lengths,
            },
        }

def joint_bros_decoder(
    hidden,
    targets,
    n_targets,
    config,
    pad_id,
    train=False,
    reuse=None,
    lengths=None,
    use_crf=True,
    **kwargs
):
    bros_targets, seq_targets = None, None
    if targets is not None:
        bros_targets = targets[:, 1:, :]
        seq_targets = targets[:, 0, :]
        
    bros_dict = bros_decoder(
        hidden,
        bros_targets,
        n_targets,
        config,
        pad_id,
        train=train,
        reuse=reuse,
        lengths=lengths,
        **kwargs
    )
    seq_dict = sequence_labeler(
        hidden,
        seq_targets,
        n_targets,
        config,
        pad_id,
        multilabel=False,
        train=train,
        reuse=reuse,
        lengths=lengths,
        use_crf=use_crf,
        **kwargs
    )
    return {
        "logits": {
            "ner_logits": seq_dict["logits"],
            "start_token_logits": bros_dict["logits"]["start_token_logits"],
            "next_token_logits": bros_dict["logits"]["next_token_logits"],
        },
        "losses": bros_dict["losses"] + seq_dict["losses"],
        "predict_params": {
            "sequence_length": lengths,
            "transition_matrix": seq_dict["predict_params"]["transition_matrix"],
        },
    }

def token_relation_decoder(
    hidden,
    targets,
    n_targets,
    config,
    pad_id,
    train=False,
    reuse=None,
    lengths=None,
    **kwargs
):
    """
    A token level relation decoder.

    :param hidden: The output of the featurizer. [batch_size, sequence_length, embed_dim]
    :param targets: The placeholder representing the targets.  [batch_size, 2, sequence_length]
    :param n_targets: A python int containing the number of classes that the model should be learning to predict over.
    :param config: A config object, containing all parameters for the featurizer.
    :param train: If this flag is true, dropout and losses are added to the graph.
    :param reuse: Should reuse be set within this scope.
    :param lengths: The number of non-padding tokens in the input.
    :param kwargs: Spare arguments.
    :return: dict containing:
        "logits": The un-normalised log probabilities of each class being in each location. For usable predictions,
            sampling from this distribution is not sufficient and a viterbi decoding method should be used.
        "losses": The negative log likelihood for the sequence targets.
        "predict_params": A dictionary of params to be fed to the viterbi decode function.
    """
    with tf.compat.v1.variable_scope("token-relation-decoder", reuse=reuse):

        if targets is not None:
            targets = tf.cast(targets, dtype=tf.int32)

        nx = config.n_embed
        hidden_size = 256

        def get_out_hidden(hidden):
            flat_logits = tf.compat.v1.layers.dense(hidden, hidden_size)
            logits_shape = tf.concat([tf.shape(hidden)[:2], [hidden_size]], 0)
            logits = tf.reshape(flat_logits, logits_shape)
            return logits

        with tf.compat.v1.variable_scope("token_hidden"):
            if config.low_memory_mode and train:
                get_out_hidden = recompute_grad(get_out_hidden, use_entire_scope=True)
            # [Batch Size, Sequence Length, Hidden Size]
            token_hidden = get_out_hidden(hidden)
            token_hidden = tf.cast(token_hidden, tf.float32)
        with tf.compat.v1.variable_scope("logits"):
            # [Batch Size, Sequence Length, Sequence Legth]
            logits = tf.matmul(
                token_hidden,
                token_hidden,
                transpose_b=True
            )

        if lengths is None:
            lengths = tf.shape(input=hidden)[1] * tf.ones(
                tf.shape(input=hidden)[0], dtype=tf.int32
            )

        loss = 0.0
        with tf.device("CPU:0" if train else logits.device):
            if targets is not None:
                targets, mask = targets[:, 0, :], targets[:, 1, :]
                weights = tf.math.divide_no_nan(
                    tf.sequence_mask(
                        lengths,
                        maxlen=tf.shape(input=targets)[1],
                        dtype=tf.float32,
                    ),
                    tf.expand_dims(tf.cast(lengths, tf.float32), -1),
                )
                targets *= mask
                logits *= mask
                bce = tf.keras.losses.BinaryCrossentropy()
                loss = bce(targets, logits, sample_weight=weights)

        return {
            "logits": logits,
            "losses": loss,
            "predict_params": {
                "entity_mask": mask,
                "sequence_length": lengths,
            },
        }

def association(
    hidden, lengths, targets, n_targets, config, train=False, reuse=None, **kwargs
):
    """
    An Attention based sequence labeler model with association.

    :param hidden: The output of the featurizer. [batch_size, sequence_length, embed_dim]
    :param lengths: The number of non-padding tokens in the input.
    :param targets: A dict containing:
     'labels': The sequence labeling targets. [batch_size, sequence_length],
     'associations': A matrix of class ids for the associations [batch_size, sequence_length, seqence_length]
    :param n_targets: A python int containing the number of classes that the model should be learning to predict over.
    :param config: A config object, containing all parameters for the featurizer.
    :param train: If this flag is true, dropout and losses are added to the graph.
    :param reuse: Should reuse be set within this scope.
    :param kwargs: Spare arguments.
    :return: dict containing:
        "logits": The un-normalised log probabilities of each class being in each location. For usable predictions,
            sampling from this distrobution is not sufficiant and a viterbi decoding method should be used.
        "losses": The negative log likelihood for the sequence targets.
        "predict_params": A dictionary of params to be fed to the viterbi decode function.
    """
    with tf.compat.v1.variable_scope("sequence-labeler", reuse=reuse):
        nx = config.n_embed
        length = config.max_length
        num_associations = len(config.association_types) + 1

        def seq_lab_internal(hidden):
            attn_fn = functools.partial(
                attn,
                scope="seq_label_attn",
                n_state=nx,
                n_head=config.seq_num_heads,
                resid_pdrop=config.resid_p_drop,
                attn_pdrop=config.attn_p_drop,
                train=train,
                scale=False,
                mask=False,
                lengths=lengths,
            )
            n = norm(attn_fn(hidden) + hidden, "seq_label_residual")
            flat_logits = tf.compat.v1.layers.dense(n, n_targets)
            logits = tf.reshape(
                flat_logits, tf.concat([tf.shape(input=hidden)[:2], [n_targets]], 0)
            )

            association_head = tf.compat.v1.layers.dense(n, nx)
            association_head = tf.reshape(
                association_head, tf.concat([tf.shape(input=hidden)[:2], [nx]], 0)
            )

            a = tf.expand_dims(association_head, 1)
            b = tf.expand_dims(association_head, 2)

            features = tf.concat(
                [
                    a - b,
                    a * b,
                    tf.tile(a, [1, length, 1, 1]),
                    tf.tile(b, [1, 1, length, 1]),
                    # TODO: Think about using prediction as a feature for associations.
                ],
                axis=-1,
            )
            associations_flat = tf.compat.v1.layers.dense(
                tf.reshape(features, shape=[-1, nx * 4]), num_associations
            )
            associations = tf.reshape(
                associations_flat, [-1, length, length, num_associations]
            )

            return logits, associations_flat, associations

        with tf.compat.v1.variable_scope("seq_lab_attn"):
            if config.low_memory_mode and train:
                seq_lab_internal = recompute_grad(
                    seq_lab_internal, use_entire_scope=True
                )

            logits, associations_flat, associations = seq_lab_internal(hidden)

        log_likelihood = 0.0
        association_loss = 0.0
        class_weights = kwargs.get("class_weights")
        if class_weights is not None:
            logits = class_reweighting(class_weights)(logits)

        transition_params = tf.compat.v1.get_variable(
            "Transition_matrix", shape=[n_targets, n_targets]
        )
        if targets is not None:
            log_likelihood, _ = crf_log_likelihood(
                logits,
                targets["labels"],
                kwargs.get("max_length")
                * tf.ones(tf.shape(input=targets["labels"])[0]),
                transition_params=transition_params,
            )
            sequence_mask = tf.sequence_mask(lengths, maxlen=length, dtype=tf.float32)
            mask = tf.expand_dims(sequence_mask, 1) * tf.expand_dims(sequence_mask, 2)

            association_loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(
                logits=associations_flat,
                labels=tf.reshape(targets["associations"], shape=[-1]),
                weights=tf.reshape(mask, shape=[-1]),
            )

        return {
            "logits": {"sequence": logits, "association": associations},
            "losses": -log_likelihood
            + config.assocation_loss_weight
            * association_loss,  # TODO: think about weighting.
            "predict_params": {"transition_matrix": transition_params},
        }
