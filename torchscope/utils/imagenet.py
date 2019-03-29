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

        for index, (synset_id, class_name) in data.items():
            key = '{}-{}'.format(synset_id.lower(), class_name.lower())
            self._index[key] = int(index)

    def __getitem__(self, target_class):
        if type(target_class) != str:
            raise TypeError('Target class needs to be a string.')

        target_class = target_class.lower()

        matches = [target_class in key for key in self.keys()]

        if not any(matches):
            raise ValueError('Cannot find the specified class.')
        elif matches.count(True) > 1:
            raise ValueError('Multiple matches found.')

        return matches.index(True)

    def __contains__(self, target_class):
        return any(target_class in key for key in self._index)

    def keys(self):
        # Don't expose synset id
        return [k.split('-')[1].replace('_', ' ') for k in self._index]
