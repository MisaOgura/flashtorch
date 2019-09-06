
import inspect
import pytest

from sys import stdout

import numpy as np
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


def test_default_lr(g_ascent):
    assert g_ascent.lr == 0.01


def test_set_custom_lr(g_ascent):
    g_ascent.lr = 0.1
    assert g_ascent.lr == 0.1


def test_default_weight_decay(g_ascent):
    assert g_ascent.weight_decay == 1e-5


def test_set_custom_weight_decay(g_ascent):
    g_ascent.weight_decay = 1e-3
    assert g_ascent.weight_decay == 1e-3


def test_default_img_size(g_ascent):
    default_img_size = 128
    assert g_ascent.img_size == default_img_size

    output = g_ascent.optimize(0, 0, 2)

    assert output.shape == (1, 3, default_img_size, default_img_size)


def test_set_custom_img_size(g_ascent):
    custom_img_size = 64
    g_ascent.img_size = custom_img_size
    assert g_ascent.img_size == custom_img_size

    output = g_ascent.optimize(0, 0, 2)

    assert output.shape == (1, 3, custom_img_size, custom_img_size)


def test_optimize_without_adam(g_ascent):
    g_ascent.with_adam = False
    output = g_ascent.optimize(0, 0, 2)

    assert output.shape == (1, 3, g_ascent.img_size, g_ascent.img_size)


def test_invalid_layer_idx_not_int(g_ascent):
    with pytest.raises(TypeError):
        g_ascent.optimize('first', 0, 2)


def test_invalid_layer_idx_negative(g_ascent):
    with pytest.raises(ValueError):
        g_ascent.optimize(-1, 0, 2)


def test_invalid_layer_idx_too_large(g_ascent):
    with pytest.raises(ValueError):
        g_ascent.optimize(15, 0, 2)  # alexnet has 13 layers


def test_invalid_layer_idx_not_conv_layer(g_ascent):
    with pytest.raises(RuntimeError):
        g_ascent.optimize(1, 0, 2)  # layer index 1 is a ReLU layer


def test_invalid_filter_idx_not_int(g_ascent):
    with pytest.raises(TypeError):
        g_ascent.optimize(0, 'first', 2)


def test_invalid_filter_idx_negative(g_ascent):
    with pytest.raises(ValueError):
        g_ascent.optimize(0, -1, 2)


def test_invalid_filter_idx_too_large(g_ascent):
    with pytest.raises(ValueError):
        g_ascent.optimize(0, 70, 2)  # the first conv layer has 64 filters


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


def test_visualize_one_filter(g_ascent):
    output = g_ascent.visualize(0, 0, 2, return_output=True)

    assert output.shape == (1, 3, g_ascent.img_size, g_ascent.img_size)


def test_visualize_random_filters_from_one_layer(g_ascent):
    num_subplots = 3

    output = g_ascent.visualize(0, num_iter=2, num_subplots=num_subplots,
                                return_output=True)

    assert len(output) == num_subplots


def test_max_num_of_subplots_is_total_num_of_filters(model, g_ascent):
    num_subplots = 100

    output = g_ascent.visualize(
        0, num_iter=2, num_subplots=num_subplots, return_output=True)

    total_num_filters = model[0].out_channels

    assert len(output) == total_num_filters


def test_visualize_specified_filters_from_one_layer(g_ascent):
    filter_idxs = np.random.choice(range(64), size=5)

    output = g_ascent.visualize(0, filter_idxs, num_iter=2, return_output=True)

    assert len(output) == len(filter_idxs)


if __name__ == '__main__':
    pytest.main([__file__])
