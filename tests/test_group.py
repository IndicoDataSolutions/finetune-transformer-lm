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
        text = ("five percent (5%) \n " +
                "fifty percent (50%) \n " +
                "two percent (2%) \n " +
                "nine percent (9%) \n " +
                "three percent (3%) \n ")
        labels = [
            {'start': 0, 'end': 17, 'label': 'z', 'text': 'five percent (5%)'},
            {'start': 20, 'end': 39, 'label': 'z', 'text': 'fifty percent (50%)'},
            {'start': 42, 'end': 58, 'label': 'z', 'text': 'two percent (2%)'},
            {'start': 61, 'end': 78, 'label': 'z', 'text': 'nine percent (9%)'},
            {'start': 81, 'end': 99, 'label': 'z', 'text': 'three percent (3%)'},
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

        self.assertEquals(preds, labels)

    def test_pipeline_tagging(self):
        model = PipelineSequenceLabeler()
        text = ("five percent (5%) \n " +
                "fifty percent (50%) \n " +
                "two percent (2%) \n " +
                "nine percent (9%) \n " +
                "three percent (3%) \n ")
        labels = [
            {'start': 0, 'end': 17, 'label': 'z', 'text': 'five percent (5%)'},
            {'start': 20, 'end': 39, 'label': 'z', 'text': 'fifty percent (50%)'},
            {'start': 42, 'end': 58, 'label': 'z', 'text': 'two percent (2%)'},
            {'start': 61, 'end': 78, 'label': 'z', 'text': 'nine percent (9%)'},
            {'start': 81, 'end': 99, 'label': 'z', 'text': 'three percent (3%)'},
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
        self.assertEquals(preds, labels[1])
