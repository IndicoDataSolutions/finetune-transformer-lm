from collections import Counter

import numpy as np
import tensorflow as tf

from finetune.encoding.input_encoder import tokenize_context
from finetune.target_models.sequence_labeling import (
    SequencePipeline,
    SequenceLabeler,
)
from finetune.encoding.target_encoders import (
    SequenceLabelingEncoder,
    GroupSequenceLabelingEncoder,
    MultiCRFGroupSequenceLabelingEncoder,
    PipelineSequenceLabelingEncoder,
)
from finetune.nn.target_blocks import (
    multi_crf_group_labeler,
    multi_logit_group_labeler,
)
from finetune.nn.crf import sequence_decode

class GroupingPipeline(SequencePipeline):
    def text_to_tokens_mask(self, X, Y=None, context=None):
        pad_token = (
            [self.config.pad_token] if self.multi_label else self.config.pad_token
        )
        out_gen = self._text_to_ids(X, pad_token=pad_token)

        for out in out_gen:
            feats = {"tokens": out.token_ids}
            if context is not None:
                tokenized_context = tokenize_context(context, out, self.config)
                feats["context"] = tokenized_context
            if Y is None:
                yield feats
            if Y is not None:
                yield feats, self.label_encoder.transform(out, Y)

class NestedPipeline(GroupingPipeline):
    def _target_encoder(self):
        return GroupSequenceLabelingEncoder(pad_token=self.config.pad_token,
                                            bio_tagging=self.config.bio_tagging)

class MultiCRFPipeline(GroupingPipeline):
    def _target_encoder(self):
        return MultiCRFGroupSequenceLabelingEncoder(pad_token=self.config.pad_token,
                                                    bio_tagging=self.config.bio_tagging)

    def feed_shape_type_def(self):
        TS = tf.TensorShape
        types = {"tokens": tf.int32}
        shapes = {"tokens": TS([None])}
        types, shapes = self._add_context_info_if_present(types, shapes)
        target_shape = [2, None]
        return (
            (
                types,
                tf.float32,
            ),
            (
                shapes,
                TS(target_shape),
            ),
        )

class PipelinePipeline(GroupingPipeline):
    def __init__(self, config, multi_label, group=True):
        super().__init__(config, multi_label)
        self.group = group

    def _target_encoder(self):
        return PipelineSequenceLabelingEncoder(pad_token=self.config.pad_token,
                                               bio_tagging=self.config.bio_tagging,
                                               group=self.group)

class GroupSequenceLabeler(SequenceLabeler):
    defaults = {"group_bio_tagging": True, "bio_tagging": True}

    def _get_input_pipeline(self):
        return NestedPipeline(
            config=self.config,
            multi_label=self.config.multi_label_sequences,
        )

    def _predict(
        self, zipped_data, per_token=False, return_negative_confidence=False, **kwargs
    ):
        _subtoken_predictions = self.config.subtoken_predictions
        self.config.subtoken_predictions = True
        annotations = super()._predict(zipped_data, per_token=per_token,
                                       return_negative_confidence=return_negative_confidence,
                                       **kwargs);
        self.config.subtoken_predictions = _subtoken_predictions

        all_groups = []
        for data, labels in zip(zipped_data, annotations):
            groups = []
            text = data["X"]
            for label in labels:
                if label["label"][:3] != "BG-" and label["label"][:3] != "IG-":
                    continue
                pre, tag = label["label"][:3], label["label"][3:]
                label["label"] = tag
                if (
                    (not groups) or
                    (pre == "BG-")
                    # (pre == "BG-") or
                    # (text[groups[-1]["tokens"][-1]["end"]:label["start"]].strip())
                ):
                    groups.append({
                        "tokens": [
                            {
                                "start": label["start"],
                                "end": label["end"],
                                "text": label["text"],
                            }
                        ],
                        "label": None
                    })
                else:
                    last_token = groups[-1]["tokens"][-1]
                    last_token["end"] = label["end"]
                    last_token["text"] = text[last_token["start"]:last_token["end"]]
            all_groups.append(groups)
        return list(zip(annotations, all_groups))


class MultiCRFGroupSequenceLabeler(GroupSequenceLabeler):
    def _get_input_pipeline(self):
        return MultiCRFPipeline(
            config=self.config,
            multi_label=self.config.multi_label_sequences,
        )

    def _compute_class_counts(self, encoded_dataset):
        counter = Counter()
        for doc, target_arr in encoded_dataset:
            target_arr = np.asarray(target_arr)
            decoded_targets = self.label_encoder.inverse_transform(target_arr,
                                                                   only_labels=True)
            counter.update(decoded_targets)
        return counter

    def _target_model(
        self,
        *,
        config,
        featurizer_state,
        targets,
        n_outputs,
        train=False,
        reuse=None,
        **kwargs
    ):
        return multi_crf_group_labeler(
            hidden=featurizer_state["sequence_features"],
            targets=targets,
            n_targets=n_outputs,
            pad_id=config.pad_idx,
            config=config,
            train=train,
            reuse=reuse,
            lengths=featurizer_state["lengths"],
            use_crf=self.config.crf_sequence_labeling,
            **kwargs
        )

    def _predict_op(self, logits, **kwargs):
        trans_mats = kwargs.get("transition_matrix")
        group_trans_mats = kwargs.get("group_transition_matrix")
        sequence_length = kwargs.get("sequence_length")
        if self.config.use_gpu_crf_predict.lower() == "auto":
            use_gpu_op = False
        else:
            use_gpu_op = self.config.use_gpu_crf_predict

        with tf.compat.v1.variable_scope("tag_sequence_decode"):
            idxs, probas = sequence_decode(
                logits[0],
                trans_mats,
                sequence_length,
                use_gpu_op=False,
                use_crf=self.config.crf_sequence_labeling,
            )
        with tf.compat.v1.variable_scope("group_sequence_decode"):
            group_idxs, group_probas = sequence_decode(
                logits[1],
                group_trans_mats,
                sequence_length,
                use_gpu_op=False,
                use_crf=self.config.crf_sequence_labeling,
            )

        # Produces [batch_size, 2, seq_len]
        idxs = tf.stack([idxs, group_idxs], axis=1)

        # Broadcast probabilities to make [batch, seq_len, n_classes * 3] matrix
        batch_seq_shape, n_classes = tf.shape(probas)[:2], tf.shape(probas)[-1]
        final_shape = tf.concat((batch_seq_shape, [n_classes * 3]), 0)
        # [batch, seq_len, n_classes, 3]
        probas = tf.expand_dims(probas, 3) * tf.expand_dims(group_probas, 2)
        # [batch, seq_len, n_classes * 3]
        probas = tf.reshape(probas, final_shape)

        return idxs, probas

class MultiLogitGroupSequenceLabeler(GroupSequenceLabeler):
    def _target_model(
        self,
        *,
        config,
        featurizer_state,
        targets,
        n_outputs,
        train=False,
        reuse=None,
        **kwargs
    ):
        return multi_logit_group_labeler(
            hidden=featurizer_state["sequence_features"],
            targets=targets,
            n_targets=n_outputs,
            pad_id=config.pad_idx,
            config=config,
            train=train,
            reuse=reuse,
            lengths=featurizer_state["lengths"],
            use_crf=self.config.crf_sequence_labeling,
            **kwargs
        )

class PipelineSequenceLabeler(SequenceLabeler):
    defaults = {"bio_tagging": True}
    def __init__(self, group=True, **kwargs):
        self.group = group
        super().__init__(**kwargs)

    def _get_input_pipeline(self):
        return PipelinePipeline(
            config=self.config,
            multi_label=self.config.multi_label_sequences,
            group=self.group
        )

    def _predict(
        self, zipped_data, per_token=False, return_negative_confidence=False, **kwargs
    ):
        """
        Transform NER span labels to group format
        """
        annotations = super()._predict(zipped_data, per_token=per_token,
                                       return_negative_confidence=return_negative_confidence,
                                       **kwargs);
        if not self.group:
            return annotations

        all_groups = []
        for labels in annotations:
            groups = []
            for label in labels:
                groups.append({
                    "tokens": [{
                        "start": label["start"],
                        "end": label["end"],
                        "text": label["text"],
                    }],
                    "label": None
                })
            all_groups.append(groups)
        return all_groups
