
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
    output = g_ascent.optimize(0, 0, 2)

    assert output.shape == (1, 3, 224, 224)


def test_optimize_with_custom_input_size(g_ascent):
    custom_input_size = 64
    output = g_ascent.optimize(0, 0, 2, size=custom_input_size)

    assert output.shape == (1, 3, custom_input_size, custom_input_size)


def test_optimize_without_adam(g_ascent):
    output = g_ascent.optimize(0, 0, 2, with_adam=False)

    assert output.shape == (1, 3, 224, 224)


def test_invalid_layer_idx_not_int(g_ascent):
    with pytest.raises(TypeError) as err:
        g_ascent.optimize('first', 0, 2)

    assert 'must be int' in str(err.value)


def test_invalid_layer_idx_negative(g_ascent):
    with pytest.raises(ValueError) as err:
        g_ascent.optimize(-1, 0, 2)

    assert 'must be zero or positive int' in str(err.value)


def test_invalid_layer_idx_too_large(g_ascent):
    with pytest.raises(ValueError) as err:
        g_ascent.optimize(15, 0, 2)  # alexnet has 13 layers

    assert 'Layer index must be <=' in str(err.value)


def test_invalid_layer_idx_not_conv_layer(g_ascent):
    with pytest.raises(RuntimeError) as err:
        g_ascent.optimize(1, 0, 2)  # layer index 1 is a ReLU layer

    assert 'is not Conv2d' in str(err.value)


def test_invalid_filter_idx_not_int(g_ascent):
    with pytest.raises(TypeError) as err:
        g_ascent.optimize(0, 'first', 2)

    assert 'must be int' in str(err.value)


def test_invalid_filter_idx_negative(g_ascent):
    with pytest.raises(ValueError) as err:
        g_ascent.optimize(0, -1, 2)

    assert 'must be zero or positive int' in str(err.value)


def test_invalid_filter_idx_too_large(g_ascent):
    with pytest.raises(ValueError) as err:
        g_ascent.optimize(0, 70, 2)  # the first conv layer has 64 filters

    assert 'Filter index must be <=' in str(err.value)


def test_register_forward_hook_to_target_layer(mocker, model):
    layer_idx = 6
    target_layer = model[layer_idx]
    mocker.spy(target_layer, 'register_forward_hook')

    g_ascent = GradientAscent(model)

    g_ascent.optimize(layer_idx, 0, 2)

    target_layer.register_forward_hook.assert_called_once()


def test_register_backward_hook_to_first_conv_layer(mocker, model):
    conv_layer = find_first_conv_layer(model, nn.modules.conv.Conv2d, 3)
    mocker.spy(conv_layer, 'register_backward_hook')

    g_ascent = GradientAscent(model)

    g_ascent.optimize(0, 0, 2)

    conv_layer.register_backward_hook.assert_called_once()


def test_visualize_one_filter(model):
    g_ascent = GradientAscent(model)
    output = g_ascent.visualize_filter(0, 0, 2, return_output=True)

    assert output.shape == (1, 3, 224, 224)


def test_visualize_one_layer(model):
    g_ascent = GradientAscent(model)

    num_subplots = 3
    output = g_ascent.visualize_layer(
        0, num_iter=2, num_subplots=num_subplots, return_output=True)

    assert len(output) == num_subplots


if __name__ == '__main__':
    pytest.main([__file__])
