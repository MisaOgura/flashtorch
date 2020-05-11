#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
"""

import json

from collections.abc import Mapping
from importlib_resources import path

from . import resources


class ImageNetIndex(Mapping):
    """Interface to retrieve ImageNet class indeces from class names.

    This class implements a dictionary like object, aiming to provide an
    easy-to-use look-up table for finding a target class index from an ImageNet
    class name.

    Reference:
        - ImageNet class index: https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json
        - Synsets: http://image-net.org/challenges/LSVRC/2015/browse-synsets

    Note:
        Class names in `imagenet_class_index.json` has been slightly modified
        from the source due to duplicated class names (e.g. crane). This helps
        make the use of this tool simpler.
    """ # noqa

    def __init__(self):
        self._index = {}

        with path(resources, 'imagenet_class_index.json') as source_path:
            with open(str(source_path), 'r') as source:
                data = json.load(source)

        for index, (_, class_name) in data.items():
            class_name = class_name.lower().replace('_', ' ')
            self._index[class_name] = int(index)

    def __len__(self):
        return len(self._index)

    def __iter__(self):
        return iter(self._index)

    def __getitem__(self, phrase):
        if type(phrase) != str:
            raise TypeError('Target class needs to be a string.')

        if phrase in self._index:
            return self._index[phrase]

        partial_matches = self._find_partial_matches(phrase)

        if not any(partial_matches):
            return None
        elif len(partial_matches) > 1:
            raise ValueError('Multiple potential matches found: {}'
                             .format(', '.join(map(str, partial_matches))))

        target_class = partial_matches.pop()

        return self._index[target_class]

    def __contains__(self, key):
        return any(key in name for name in self._index)

    def keys(self):
        return self._index.keys()

    def items(self):
        return self._index.items()

    def _find_partial_matches(self, phrase):
        words = phrase.lower().split(' ')

        # Find the intersection between search words and class names to
        # prioritise whole word matches
        # e.g. If words = {'dalmatian', 'dog'} then matches 'dalmatian'

        matches = set(words).intersection(set(self.keys()))

        if not any(matches):
            # Find substring matches between search words and class names to
            # accommodate for fuzzy matches to some extend
            # e.g. If words = {'foxhound'} then matches 'english foxhound'

            matches = [key for word in words for key in self.keys()
                       if word in key]

        return matches
