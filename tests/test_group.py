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
)

class TestGroupingLabelers(unittest.TestCase):
    def test_nested_tagging(self):
        model = GroupSequenceLabeler()
        text = "five percent ? (5%)"
        labels = [
            {'start': 0, 'end': 4, 'label': 'z', 'text': 'five'},
            {'start': 5, 'end': 14, 'label': 'z', 'text': 'percent ?'},
            {'start': 15, 'end': 19, 'label': 'z', 'text': '(5%)'},
        ]
        groups = [
            {'tokens': [
                {'start': 5, 'end': 19, 'text': 'percent ? (5%)'},
            ], 'label': None}
        ]
        labels = (labels, groups)
        model.fit([text] * 30, [labels] * 30)
        preds = model.predict([text])[0]
        self.assertEqual(len(preds), 2)
        self.assertEqual(len(preds[0]), 3)
        self.assertEqual(len(preds[1]), 1)
        for p in preds[0]:
            del p["confidence"]
        self.assertEquals(preds, labels)

    def test_pipeline_tagging(self):
        model = PipelineSequenceLabeler()
        text = "five percent ? (5%)"
        labels = [
            {'start': 0, 'end': 4, 'label': 'z', 'text': 'five'},
            {'start': 5, 'end': 14, 'label': 'z', 'text': 'percent ?'},
            {'start': 15, 'end': 19, 'label': 'z', 'text': '(5%)'},
        ]
        groups = [
            {'tokens': [
                {'start': 5, 'end': 19, 'text': 'percent ? (5%)'},
            ], 'label': None}
        ]
        labels = (labels, groups)
        model.fit([text] * 30, [labels] * 30)
        preds = model.predict([text])[0]
        self.assertEqual(len(preds), 1)
        self.assertEquals(preds, labels[1])
