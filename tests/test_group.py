import os
import unittest
import logging
from copy import copy
from pathlib import Path
import codecs
import json
import random
import time
# required for tensorflow logging control
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pytest
from pytest import approx

import tensorflow as tf
import numpy as np

from finetune.target_models.grouping import (
    GroupSequenceLabeler,
    PipelineSequenceLabeler,
    MultiCRFGroupSequenceLabeler,
    MultiLogitGroupSequenceLabeler,
    BROSLabeler,
)

class TestGroupingLabelers(unittest.TestCase):
    def test_nested_tagging(self):
        model = GroupSequenceLabeler(class_weights="sqrt")
        text = ("five percent (5%) \n " +
                "fifty percent (50%) \n " +
                "two percent (2%) \n " +
                "nine percent (9%) \n " +
                "three percent (3%) \n ")
        labels = [
            {'start': 0, 'end': 17, 'label': 'a', 'text': 'five percent (5%)'},
            {'start': 20, 'end': 39, 'label': 'b', 'text': 'fifty percent (50%)'},
            {'start': 42, 'end': 58, 'label': 'a', 'text': 'two percent (2%)'},
            {'start': 61, 'end': 78, 'label': 'b', 'text': 'nine percent (9%)'},
            {'start': 81, 'end': 99, 'label': 'a', 'text': 'three percent (3%)'},
        ]
        groups = [
            {'tokens': [
                {'start': 0, 'end': 39, 'text': 'five percent (5%) \n fifty percent (50%)'},
            ], 'label': None},
            {'tokens': [
                {'start': 61, 'end': 99, 'text': 'nine percent (9%) \n three percent (3%)'},
            ], 'label': None}
        ]
        labels = (labels, groups)

        model.fit([text] * 30, [labels] * 30)
        preds = model.predict([text])[0]

        self.assertEqual(len(preds), 2)
        self.assertEqual(len(preds[0]), 5)
        self.assertEqual(len(preds[1]), 2)

        for p in preds[0]:
            del p["confidence"]

        self.assertEqual(preds, labels)

    def test_multi_crf_tagging(self):
        model = MultiCRFGroupSequenceLabeler(crf_sequence_labeling=True,
                                             class_weights="sqrt")
        text = ("five percent (5%) \n " +
                "fifty percent (50%) \n " +
                "two percent (2%) \n " +
                "nine percent (9%) \n " +
                "three percent (3%) \n ")
        labels = [
            {'start': 0, 'end': 17, 'label': 'a', 'text': 'five percent (5%)'},
            {'start': 20, 'end': 39, 'label': 'b', 'text': 'fifty percent (50%)'},
            {'start': 42, 'end': 58, 'label': 'a', 'text': 'two percent (2%)'},
            {'start': 61, 'end': 78, 'label': 'b', 'text': 'nine percent (9%)'},
            {'start': 81, 'end': 99, 'label': 'a', 'text': 'three percent (3%)'},
        ]
        groups = [
            {'tokens': [
                {'start': 0, 'end': 39, 'text': 'five percent (5%) \n fifty percent (50%)'},
            ], 'label': None},
            {'tokens': [
                {'start': 61, 'end': 99, 'text': 'nine percent (9%) \n three percent (3%)'},
            ], 'label': None}
        ]
        labels = (labels, groups)

        model.fit([text] * 30, [labels] * 30)
        preds = model.predict([text])[0]

        self.assertEqual(len(preds), 2)
        self.assertEqual(len(preds[0]), 5)
        self.assertEqual(len(preds[1]), 2)

        for p in preds[0]:
            del p["confidence"]

        self.assertEqual(preds, labels)

    def test_multi_logit_tagging(self):
        model = MultiLogitGroupSequenceLabeler(crf_sequence_labeling=True,
                                             class_weights="sqrt")
        text = ("five percent (5%) \n " +
                "fifty percent (50%) \n " +
                "two percent (2%) \n " +
                "nine percent (9%) \n " +
                "three percent (3%) \n ")
        labels = [
            {'start': 0, 'end': 17, 'label': 'a', 'text': 'five percent (5%)'},
            {'start': 20, 'end': 39, 'label': 'b', 'text': 'fifty percent (50%)'},
            {'start': 42, 'end': 58, 'label': 'a', 'text': 'two percent (2%)'},
            {'start': 61, 'end': 78, 'label': 'b', 'text': 'nine percent (9%)'},
            {'start': 81, 'end': 99, 'label': 'a', 'text': 'three percent (3%)'},
        ]
        groups = [
            {'tokens': [
                {'start': 0, 'end': 39, 'text': 'five percent (5%) \n fifty percent (50%)'},
            ], 'label': None},
            {'tokens': [
                {'start': 61, 'end': 99, 'text': 'nine percent (9%) \n three percent (3%)'},
            ], 'label': None}
        ]
        labels = (labels, groups)

        model.fit([text] * 30, [labels] * 30)
        preds = model.predict([text])[0]

        self.assertEqual(len(preds), 2)
        self.assertEqual(len(preds[0]), 5)
        self.assertEqual(len(preds[1]), 2)

        for p in preds[0]:
            del p["confidence"]

        self.assertEqual(preds, labels)

    def test_pipeline_tagging(self):
        model = PipelineSequenceLabeler(class_weights="sqrt")
        text = ("five percent (5%) \n " +
                "fifty percent (50%) \n " +
                "two percent (2%) \n " +
                "nine percent (9%) \n " +
                "three percent (3%) \n ")
        labels = [
            {'start': 0, 'end': 17, 'label': 'a', 'text': 'five percent (5%)'},
            {'start': 20, 'end': 39, 'label': 'b', 'text': 'fifty percent (50%)'},
            {'start': 42, 'end': 58, 'label': 'a', 'text': 'two percent (2%)'},
            {'start': 61, 'end': 78, 'label': 'b', 'text': 'nine percent (9%)'},
            {'start': 81, 'end': 99, 'label': 'a', 'text': 'three percent (3%)'},
        ]
        groups = [
            {'tokens': [
                {'start': 0, 'end': 39, 'text': 'five percent (5%) \n fifty percent (50%)'},
            ], 'label': None},
            {'tokens': [
                {'start': 61, 'end': 99, 'text': 'nine percent (9%) \n three percent (3%)'},
            ], 'label': None}
        ]
        labels = (labels, groups)
        model.fit([text] * 30, [labels] * 30)
        preds = model.predict([text])[0]
        self.assertEqual(len(preds), 2)
        self.assertEqual(preds, labels[1])

    def test_bros_tagging(self):
        model = BROSLabeler()
        text = ("five percent (5%) \n " +
                "fifty percent (50%) \n " +
                "two percent (2%) \n " +
                "nine percent (9%) \n " +
                "three percent (3%) \n ")
        labels = [
            {'start': 0, 'end': 17, 'label': 'a', 'text': 'five percent (5%)'},
            {'start': 20, 'end': 39, 'label': 'b', 'text': 'fifty percent (50%)'},
            {'start': 42, 'end': 58, 'label': 'a', 'text': 'two percent (2%)'},
            {'start': 61, 'end': 78, 'label': 'b', 'text': 'nine percent (9%)'},
            {'start': 81, 'end': 99, 'label': 'a', 'text': 'three percent (3%)'},
        ]
        groups = [
            {'tokens': [
                {'start': 0, 'end': 39, 'text': 'five percent (5%) \n fifty percent (50%)'},
            ], 'label': None},
            {'tokens': [
                {'start': 61, 'end': 99, 'text': 'nine percent (9%) \n three percent (3%)'},
            ], 'label': None}
        ]
        labels = (labels, groups)
        model.fit([text] * 30, [labels] * 30)
        preds = model.predict([text])[0]
        print(preds)
        self.assertEqual(len(preds), 2)
        self.assertEqual(preds, labels[1])
