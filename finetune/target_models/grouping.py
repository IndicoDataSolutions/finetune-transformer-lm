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
    def __init__(self, config, multi_label, nested_group_tagging=False,
                 pipeline_group_tagging=False):
        super().__init__(config, multi_label)
        self.nested_group_tagging = nested_group_tagging
        self.pipeline_group_tagging = pipeline_group_tagging

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

    def _target_encoder(self):
        if self.nested_group_tagging:
            return GroupSequenceLabelingEncoder(pad_token=self.config.pad_token,
                                           bio_tagging=self.config.bio_tagging)
        if self.pipeline_group_tagging:
            return PipelineSequenceLabelingEncoder(pad_token=self.config.pad_token,
                                           bio_tagging=self.config.bio_tagging)
        return SequenceLabelingEncoder(pad_token=self.config.pad_token,
                                       bio_tagging=self.config.bio_tagging)

class GroupSequenceLabeler(SequenceLabeler):
    defaults = {"group_bio_tagging": True, "bio_tagging": True}
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_input_pipeline(self):
        return GroupingPipeline(
            config=self.config,
            multi_label=self.config.multi_label_sequences,
            nested_group_tagging=True
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


class PipelineSequenceLabeler(SequenceLabeler):
    defaults = {"bio_tagging": True}
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_input_pipeline(self):
        return GroupingPipeline(
            config=self.config,
            multi_label=self.config.multi_label_sequences,
            pipeline_group_tagging=True
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
        all_groups = []
        for labels in annotations:
            groups = []
            for label in labels:
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
            all_groups.append(groups)
        return all_groups
