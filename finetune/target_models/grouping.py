import warnings
from collections import Counter

import numpy as np
import tensorflow as tf

from finetune.model import PredictMode
from finetune.nn.crf import sequence_decode
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
    BROSEncoder,
    JointBROSEncoder,
    TokenRelationEncoder,
)
from finetune.nn.target_blocks import (
    multi_crf_group_labeler,
    multi_logit_group_labeler,
    bros_decoder,
    joint_bros_decoder,
    token_relation_decoder,
)

class GroupingPipeline(SequencePipeline):
    def __init__(self, config):
        super().__init__(config, multi_label=False)

    def text_to_tokens_mask(self, X, Y=None, context=None):
        pad_token = self.config.pad_token
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
    def __init__(self, config, group=True):
        super().__init__(config)
        self.group = group

    def _target_encoder(self):
        return PipelineSequenceLabelingEncoder(pad_token=self.config.pad_token,
                                               bio_tagging=self.config.bio_tagging,
                                               group=self.group)

class BROSPipeline(GroupingPipeline):
    def _target_encoder(self):
        return BROSEncoder(pad_token=self.config.pad_token)

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

class JointBROSPipeline(GroupingPipeline):
    def _target_encoder(self):
        return JointBROSEncoder(pad_token=self.config.pad_token,
                                bio_tagging=self.config.bio_tagging)

    def feed_shape_type_def(self):
        TS = tf.TensorShape
        types = {"tokens": tf.int32}
        shapes = {"tokens": TS([None])}
        types, shapes = self._add_context_info_if_present(types, shapes)
        target_shape = [3, None]
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

class TokenRelationPipeline(GroupingPipeline):
    def _target_encoder(self):
        return TokenRelationEncoder(pad_token=self.config.pad_token)

    def feed_shape_type_def(self):
        TS = tf.TensorShape
        types = {"tokens": tf.int32}
        shapes = {"tokens": TS([None])}
        types, shapes = self._add_context_info_if_present(types, shapes)
        target_shape = [2, None, None]
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

class JointTokenRelationPipeline(GroupingPipeline):
    def _target_encoder(self):
        return JointTokenRelationEncoder(pad_token=self.config.pad_token)

    def feed_shape_type_def(self):
        TS = tf.TensorShape
        types = {"tokens": tf.int32}
        shapes = {"tokens": TS([None])}
        types, shapes = self._add_context_info_if_present(types, shapes)
        # TODO: Fix this
        target_shape = [3, None, None]
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


class GroupSequenceLabeler(SequenceLabeler):
    defaults = {"group_bio_tagging": True, "bio_tagging": True}

    def _get_input_pipeline(self):
        return NestedPipeline(
            config=self.config,
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
        # TODO: Fix this breakig when loading from model
        if not self.group:
        # if not True:
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

class BROSLabeler(SequenceLabeler):
    defaults = {"chunk_long_sequences": False}

    def _get_input_pipeline(self):
        return BROSPipeline(config=self.config)

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
        return bros_decoder(
            hidden=featurizer_state["sequence_features"],
            targets=targets,
            n_targets=n_outputs,
            pad_id=config.pad_idx,
            config=config,
            train=train,
            reuse=reuse,
            lengths=featurizer_state["lengths"],
            **kwargs
        )

    def _predict_op(self, logits, **kwargs):
        # [Batch Size, Sequence Length, 2]
        start_token_logits = logits["start_token_logits"]
        start_token_idxs = tf.argmax(start_token_logits, axis=-1)
        start_token_probas = tf.nn.softmax(start_token_logits, axis=-1)

        # [Batch Size, Sequence Length, Sequence Length + 1]
        next_token_logits = logits["next_token_logits"]
        next_token_idxs = tf.argmax(next_token_logits, axis=-1)
        next_token_probas = tf.nn.softmax(next_token_logits, axis=-1)

        # idxs = (start_token_idxs, next_token_idxs)

        # Produces [batch_size, 2, seq_len]
        idxs = tf.stack([start_token_idxs, next_token_idxs], axis=1)
        probas = start_token_probas

        return idxs, probas

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)

    def predict_proba(self, X, **kwargs):
        raise NotImplementedError

    def _predict(self, zipped_data, **kwargs):
        predictions = list(self.process_long_sequence(zipped_data))
        return self._predict_decode(zipped_data, predictions, **kwargs)

    def _predict_decode(self, zipped_data, predictions, **kwargs):
        raw_texts = list(data.get("raw_text", data["X"]) for data in zipped_data)
        all_groups = []
        for text_idx, (
            token_start_idx,
            token_end_idx,
            start_of_doc,
            end_of_doc,
            label_seq,
            proba_seq,
            start,
            end,
        ) in enumerate(predictions):
            assert start_of_doc and end_of_doc, "Chunk found in BROS!"

            start_tokens, next_tokens = label_seq
            # start_proba, next_proba = proba_seq

            text = raw_texts[text_idx]
            doc_groups = []
            for i, (label, start_idx, end_idx) in enumerate(zip(
                start_tokens, token_start_idx, token_end_idx
            )):
                if label == self.config.pad_token:
                    continue

                group_spans = [{
                    "start": start_idx,
                    "end": end_idx,
                    "text": text[start_idx:end_idx],
                }] 

                group_idxs = [i]
                current_idx = next_tokens[i] 
                while current_idx > 0:
                    if current_idx in group_idxs:
                        warnings.warn("Cylical group found!")
                        break

                    current_start_idx = token_start_idx[current_idx]
                    current_end_idx = token_end_idx[current_idx]
                    for span in group_spans:
                        if (current_start_idx >= span["end"] and
                            not text[span["end"]:current_start_idx].strip()):
                            span["end"] = current_end_idx
                            span["text"] = text[span["start"]:span["end"]]
                            break
                        elif (current_end_idx <= span["start"] and
                              not text[current_end_idx:span["start"]].strip()):
                            span["start"] = current_start_idx
                            span["text"] = text[span["start"]:span["end"]]
                            break
                    else:
                        group_spans.append({
                            "start": current_start_idx,
                            "end": current_end_idx,
                            "text": text[current_start_idx:current_end_idx],
                        })

                    group_idxs.append(current_idx)
                    current_idx = next_tokens[current_idx]
                group_spans = sorted(group_spans, key=lambda x: x["start"])
                doc_groups.append({
                    "tokens": group_spans,
                    "label": None,
                })
            doc_groups = sorted(doc_groups, key=lambda x: x["tokens"][0]["start"])
            all_groups.append(doc_groups)
        return all_groups

class JointBROSLabeler(BROSLabeler, SequenceLabeler):
    defaults = {"chunk_long_sequences": False}

    def _get_input_pipeline(self):
        return JointBROSPipeline(config=self.config)

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
        return joint_bros_decoder(
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
        group_idxs, group_probas = BROSLabeler._predict_op(
            self, logits, **kwargs
        )
        ner_idxs, ner_probas = SequenceLabeler._predict_op(
            self, logits["ner_logits"], **kwargs
        )

        # Check why this is neccesary
        group_idxs = tf.cast(group_idxs, tf.int32)
        start_token_idxs, next_token_idxs = group_idxs[:, 0, :], group_idxs[:, 1, :]

        # Produces [batch_size, 3, seq_len]
        idxs = tf.stack([ner_idxs, start_token_idxs, next_token_idxs], axis=1)
        probas = ner_probas

        return idxs, probas

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)

    def predict_proba(self, X, **kwargs):
        raise NotImplementedError

    def _predict(self, zipped_data, **kwargs):
        # This is somewhat horrifying
        predictions = list(self.process_long_sequence(zipped_data))
        (token_start_idx, token_end_idx, start_of_doc, end_of_doc, label_seq,
         proba_seq, start, end) = list(zip(*predictions))
        ner_labels, start_token_labels, next_token_labels = list(zip(*label_seq))

        group_labels = list(zip(start_token_labels, next_token_labels))
        group_predictions = zip(token_start_idx, token_end_idx, start_of_doc,
                              end_of_doc, group_labels, proba_seq, start, end)
        ner_predictions = zip(token_start_idx, token_end_idx, start_of_doc,
                              end_of_doc, ner_labels, proba_seq, start, end)


        group_predictions = BROSLabeler._predict_decode(
            self, zipped_data, group_predictions, **kwargs
        )
        ner_predictions = SequenceLabeler._predict_decode(
            self, zipped_data, ner_predictions, **kwargs
        )
        return list(zip(ner_predictions, group_predictions))

class TokenRelationLabeler(SequenceLabeler):
    defaults = {"chunk_long_sequences": False}

    def _get_input_pipeline(self):
        return TokenRelationPipeline(config=self.config)

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
        return token_relation_decoder(
            hidden=featurizer_state["sequence_features"],
            targets=targets,
            n_targets=n_outputs,
            pad_id=config.pad_idx,
            config=config,
            train=train,
            reuse=reuse,
            lengths=featurizer_state["lengths"],
            **kwargs
        )

    def _predict_op(self, logits, **kwargs):
        # TODO: Move to config
        thresh = 0.8
        idxs = kwargs.get("entity_mask")

        probas = tf.sigmoid(logits) * mask
        relations = tf.cast(probas > thresh, int32)

        return relations, probas

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)

    def predict_proba(self, X, **kwargs):
        raise NotImplementedError

    def _predict(self, zipped_data, **kwargs):
        predictions = list(self.process_long_sequence(zipped_data))
        return self._predict_decode(zipped_data, predictions, **kwargs)

    def _predict_decode(self, zipped_data, predictions, **kwargs):
        raw_texts = list(data.get("raw_text", data["X"]) for data in zipped_data)
        all_groups = []
        for text_idx, (
            token_start_idx,
            token_end_idx,
            start_of_doc,
            end_of_doc,
            label_seq,
            proba_seq,
            start,
            end,
        ) in enumerate(predictions):
            assert start_of_doc and end_of_doc, "Chunk found in token relation!!"

            text = raw_texts[text_idx]
            doc_groups = []
        return
    
class JointTokenRelationLabeler(TokenRelationLabeler, SequenceLabeler):
    defaults = {"chunk_long_sequences": False}

    def _get_input_pipeline(self):
        return JointTokenRelationPipeline(config=self.config)

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
        return joint_token_relation_decoder(
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
        group_idxs, group_probas = TokenRelationLabeler._predict_op(
            self, logits["group_logits"], **kwargs
        )
        ner_idxs, ner_probas = SequenceLabeler._predict_op(
            self, logits["ner_logits"], **kwargs
        )

        # Check why this is neccesary
        group_idxs = tf.cast(group_idxs, tf.int32)

        # Produces [batch_size, 3, seq_len]
        # idxs = tf.stack([ner_idxs, start_token_idxs, next_token_idxs], axis=1)
        # TODO: Figure out how to do this
        probas = ner_probas

        return idxs, probas

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)

    def predict_proba(self, X, **kwargs):
        raise NotImplementedError

    def _predict(self, zipped_data, **kwargs):
        predictions = list(self.process_long_sequence(zipped_data))
        (token_start_idx, token_end_idx, start_of_doc, end_of_doc, label_seq,
         proba_seq, start, end) = list(zip(*predictions))
        ner_labels, group_labels = list(zip(*label_seq))

        group_predictions = zip(token_start_idx, token_end_idx, start_of_doc,
                              end_of_doc, group_labels, proba_seq, start, end)
        ner_predictions = zip(token_start_idx, token_end_idx, start_of_doc,
                              end_of_doc, ner_labels, proba_seq, start, end)


        group_predictions = BROSLabeler._predict_decode(
            self, zipped_data, group_predictions, **kwargs
        )
        ner_predictions = SequenceLabeler._predict_decode(
            self, zipped_data, ner_predictions, **kwargs
        )
        return list(zip(ner_predictions, group_predictions))
