import itertools
import copy
from collections import Counter

import tensorflow as tf
import numpy as np

from finetune.encoding.input_encoder import tokenize_context
from finetune.target_models.sequence_labeling import (
    SequencePipeline,
    SequenceLabeler,
)
from finetune.encoding.target_encoders import (
    SequenceLabelingEncoder,
    GroupSequenceLabelingEncoder,
    PipelineSequenceLabelingEncoder,
)

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

class GroupSequenceLabeler(SequenceLabeler):
    defaults = {"group_bio_tagging": True, "bio_tagging": True}
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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


class PipelinePipeline(GroupingPipeline):
    def __init__(self, config, multi_label, group=True):
        super().__init__(config, multi_label)
        self.group = group

    def _target_encoder(self):
        return PipelineSequenceLabelingEncoder(pad_token=self.config.pad_token,
                                               bio_tagging=self.config.bio_tagging,
                                               group=self.group)

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
