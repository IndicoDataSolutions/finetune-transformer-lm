import itertools
import copy
from collections import Counter

import tensorflow as tf
import numpy as np

from finetune.base import BaseModel
from finetune.encoding.target_encoders import (
    GroupSequenceLabelingEncoder,
    PipelineSequenceLabelingEncoder,
)
from finetune.target_models import (
    SequencePipeline,
    SequenceLabeler,
)

class GroupSequenceLabeler(SequenceLabeler):
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
        annotations = super()._predict();

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
        annotations = super()._predict();
