import itertools
import copy
from collections import Counter

import tensorflow as tf
import numpy as np

from finetune.target_models import (
    SequencePipeline,
    SequenceLabeler,
)

class GroupSequenceLabeler(SequenceLabeler):
    defaults = {"group_bio_tagging": True}
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_input_pipeline(self):
        return SequencePipeline(
            config=self.config,
            multi_label=self.config.multi_label_sequences,
            nested_group_tagging=True
        )

    def _predict(
        self, zipped_data, per_token=False, return_negative_confidence=False, **kwargs
    ):
        _subtoken_predictions = self.config.subtoken_predictions
        self.config.subtoken_predictions = True
        annotations = super()._predict();
        self.config.subtoken_predictions = _subtoken_predictions

        all_groups = []
        for labels in annotations:
            groups = []
            for label in labels:
                if label[:3] != "BG-" and label[:3] != "IG-":
                    continue
                pre, tag = label["label"][:3], label["label"][3:]
                label["label"] = tag
                if pre == "BG-":
                    groups.append({
                        "tokens": [
                            {
                                "start": label["start"],
                                "end": label["end"],
                                "test": label["text"],
                            }
                        ],
                        "label": None
                    })
                else:
                    groups[-1]["tokens"][-1]["end"] = label["end"]
                    groups[-1]["tokens"][-1]["text"] += " " + label["text"]
            all_groups.append(groups)
        return list(zip(annotations, all_groups)


class PipelineSequenceLabeler(SequenceLabeler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_input_pipeline(self):
        return SequencePipeline(
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
        annotations = super()._predict();
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
