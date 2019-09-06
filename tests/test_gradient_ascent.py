
import inspect
import pytest

from sys import stdout

import numpy as np
import torch.nn as nn
import torchvision.models as models

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
def conv_layer(model):
    return model[0]

@pytest.fixture
def available_models():
    return inspect.getmembers(models, inspect.isfunction)


@pytest.fixture
def g_ascent(model):
    return GradientAscent(model)


##############
# Test cases #
##############


def test_set_custom_lr(g_ascent):
    g_ascent.lr = 0.1
    assert g_ascent.lr == 0.1


def test_set_custom_weight_decay(g_ascent):
    g_ascent.weight_decay = 1e-3
    assert g_ascent.weight_decay == 1e-3


def test_optimize(g_ascent, conv_layer):
    default_img_size = 128
    assert g_ascent.img_size == default_img_size

    output = g_ascent.optimize(conv_layer, 0, 2)

    assert output.shape == (1, 3, default_img_size, default_img_size)


def test_set_custom_img_size(conv_layer, g_ascent):
    custom_img_size = 64
    g_ascent.img_size = custom_img_size
    assert g_ascent.img_size == custom_img_size

    output = g_ascent.optimize(conv_layer, 0, 2)

    assert output.shape == (1, 3, custom_img_size, custom_img_size)


def test_optimize_without_adam(conv_layer, g_ascent):
    g_ascent.with_adam = False
    output = g_ascent.optimize(conv_layer, 0, 2)

    assert output.shape == (1, 3, g_ascent.img_size, g_ascent.img_size)


def test_invalid_layer_str(g_ascent):
    with pytest.raises(TypeError):
        g_ascent.optimize('first', 0, 2)


def test_invalid_layer_int(g_ascent):
    with pytest.raises(TypeError):
        g_ascent.optimize(0, 0, 2)


def test_invalid_layer_not_conv(model, g_ascent):
    with pytest.raises(TypeError):
        g_ascent.optimize(model[1], 0, 2)  # model[1] is  ReLU layer


def test_invalid_filter_idx_not_int(conv_layer, g_ascent):
    with pytest.raises(TypeError):
        g_ascent.optimize(conv_layer, 'first', 2)


def test_invalid_filter_idx_negative(conv_layer, g_ascent):
    with pytest.raises(ValueError):
        g_ascent.optimize(conv_layer, -1, 2)


def test_invalid_filter_idx_too_large(conv_layer, g_ascent):
    with pytest.raises(ValueError):
        g_ascent.optimize(conv_layer, 70, 2)  # the target conv layer has 64 filters


def test_register_forward_hook_to_target_layer(mocker, conv_layer, model):
    mocker.spy(conv_layer, 'register_forward_hook')

    g_ascent = GradientAscent(model)
    g_ascent.optimize(conv_layer, 0, 2)

    conv_layer.register_forward_hook.assert_called_once()


def test_register_backward_hook_to_first_conv_layer(mocker, conv_layer, model):
    mocker.spy(conv_layer, 'register_backward_hook')

    g_ascent = GradientAscent(model)
    g_ascent.optimize(conv_layer, 0, 2)

    conv_layer.register_backward_hook.assert_called_once()


def test_visualize_one_filter(conv_layer, g_ascent):
    output = g_ascent.visualize(conv_layer, 0, 2, return_output=True)

    assert output.shape == (1, 3, g_ascent.img_size, g_ascent.img_size)


def test_visualize_random_filters_from_one_layer(conv_layer, g_ascent):
    num_subplots = 3

    output = g_ascent.visualize(conv_layer, num_iter=2,
                                num_subplots=num_subplots,
                                return_output=True)

    assert len(output) == num_subplots


def test_max_num_of_subplots_is_total_num_of_filters(conv_layer, g_ascent):
    num_subplots = 100

    output = g_ascent.visualize(
        conv_layer, num_iter=2, num_subplots=num_subplots, return_output=True)

    assert len(output) == conv_layer.out_channels


def test_visualize_specified_filters_from_one_layer(conv_layer, g_ascent):
    filter_idxs = np.random.choice(range(64), size=5)

    output = g_ascent.visualize(
        conv_layer, filter_idxs, num_iter=2, return_output=True)

    assert len(output) == len(filter_idxs)


if __name__ == '__main__':
    pytest.main([__file__])
