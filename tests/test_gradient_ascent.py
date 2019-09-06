
import inspect
import pytest

from sys import stdout

import torch.nn as nn
import torchvision.models as models

from common_utils import *
from flashtorch.activemax import GradientAscent


#####################
# Utility functions #
#####################


#################
# Test fixtures #
#################


@pytest.fixture
def model():
    return models.alexnet().features


@pytest.fixture
def available_models():
    return inspect.getmembers(models, inspect.isfunction)


@pytest.fixture
def g_ascent(model):
    return GradientAscent(model)

##############
# Test cases #
##############


def test_optimize_with_default_input_size(g_ascent):
    output = g_ascent.optimize(0, 0, 5)

    assert output.shape == (1, 3, 224, 224)


def test_optimize_with_custom_input_size(g_ascent):
    custom_input_size = 64
    output = g_ascent.optimize(0, 0, 5, size=custom_input_size)

    assert output.shape == (1, 3, custom_input_size, custom_input_size)


def test_optimize_without_adam(g_ascent):
    output = g_ascent.optimize(0, 0, 5, with_adam=False)

    assert output.shape == (1, 3, 224, 224)


def test_invalid_layer_idx_not_int(g_ascent):
    with pytest.raises(TypeError) as err:
        g_ascent.optimize('first', 0, 5)

    assert 'must be int' in str(err.value)


def test_invalid_layer_idx_negative(g_ascent):
    with pytest.raises(ValueError) as err:
        g_ascent.optimize(-1, 0, 5)

    assert 'must be zero or positive int' in str(err.value)


def test_invalid_layer_idx_too_large(g_ascent):
    with pytest.raises(ValueError) as err:
        g_ascent.optimize(15, 0, 5)  # alexnet has 13 layers

    assert 'Layer index must be <=' in str(err.value)


def test_invalid_layer_idx_not_conv_layer(g_ascent):
    with pytest.raises(RuntimeError) as err:
        g_ascent.optimize(1, 0, 5)  # layer index 1 is a ReLU layer

    assert 'is not of Conv2d' in str(err.value)


def test_invalid_filter_idx_not_int(g_ascent):
    with pytest.raises(TypeError) as err:
        g_ascent.optimize(0, 'first', 5)

    assert 'must be int' in str(err.value)


def test_invalid_filter_idx_negative(g_ascent):
    with pytest.raises(ValueError) as err:
        g_ascent.optimize(0, -1, 5)

    assert 'must be zero or positive int' in str(err.value)


def test_invalid_filter_idx_too_large(g_ascent):
    with pytest.raises(ValueError) as err:
        g_ascent.optimize(0, 70, 5)  # the first conv layer has 64 filters

    assert 'Filter index must be <=' in str(err.value)


if __name__ == '__main__':
    pytest.main([__file__])
