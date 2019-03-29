#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
"""

import pdb
import json
from os import path


class ImageNetIndex:
    """
    'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'
    """

    source = path.join(path.dirname(__file__),
                       './resources/imagenet_class_index.json')

    def __init__(self):
        self._index = {}

        with open(ImageNetIndex.source, 'r') as source:
            data = json.load(source)

        for index, (_, class_name) in data.items():
            class_name = class_name.replace('_', ' ')
            self._index[class_name] = int(index)

    def __getitem__(self, target_class):
        if type(target_class) != str:
            raise TypeError('Target class needs to be a string.')

        if target_class not in self._index:
            raise ValueError('Cannot find the specified class.')

        return self._index[target_class]

    def __contains__(self, target_class):
        return target_class in self._index

    def keys(self):
        return self._index.keys()
