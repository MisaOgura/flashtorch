#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
"""

import pdb
import json
from os import path


class ImageNetIndex:
    """
    ImageNet class index: 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'
    Synsets for 1000 classes in ImageNet: http://image-net.org/challenges/LSVRC/2015/browse-synsets

    Due to the exact same class names between crane (lifting device) and crane
    (wading bird),
    """

    source = path.join(path.dirname(__file__),
                       './resources/imagenet_class_index.json')

    def __init__(self):
        self._index = {}

        with open(ImageNetIndex.source, 'r') as source:
            data = json.load(source)

        for index, (_, class_name) in data.items():
            class_name = class_name.lower().replace('_', ' ')
            self._index[class_name] = int(index)

    def __getitem__(self, phrase):
        if type(phrase) != str:
            raise TypeError('Target class needs to be a string.')

        words = phrase.lower().split(' ')

        # Find the intersection between search words and class names to
        # prioritise whole word matches
        # e.g. words = {'dalmatian', 'dog'} then matches 'dalmatian'

        matches = set(words).intersection(set(self.keys()))

        if not any(matches):
            # Find substring matches between search words and class names to
            # accommodate for fuzzy matches to some extend
            # e.g. words = {'foxhound'} then matches 'english foxhound'

            matches = [key for word in words for key in self.keys() \
                if word in key]

            if not any(matches):
                raise ValueError('Cannot find the specified class.\n' \
                                 'See available classes with .keys()')

        if len(matches) > 1:
            raise ValueError('Multiple matches found.\n' \
                             'See the available class with .keys()')

        target_class = matches.pop()

        return self._index[target_class]

    def __contains__(self, key):
        return any(key in name for name in self._index)

    def keys(self):
        return self._index.keys()

    def items(self):
        return self._index.items()
