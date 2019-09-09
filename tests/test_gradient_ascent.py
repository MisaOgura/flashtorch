
import inspect
import pytest

from os import path
from sys import stdout

import numpy as np
import torch.nn as nn
import torchvision.models as models

from flashtorch.activmax import GradientAscent

from flashtorch.utils import apply_transforms


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
    g_ascent = GradientAscent(model)
    g_ascent.img_size = 64  # to reduce test time
    return g_ascent


##############
# Test cases #
##############


def test_optimize(g_ascent, conv_layer):
    output = g_ascent.optimize(conv_layer, 0, num_iter=2)

    assert len(output) == 2  # num_iter
    assert output[0].shape == (1, 3, g_ascent.img_size, g_ascent.img_size)


def test_optimize_with_custom_input(mocker, conv_layer, model):
    mocker.spy(model, 'forward')
    g_ascent = GradientAscent(model)

    custom_input = np.uint8(np.random.uniform(150, 180, (64, 64, 3)))
    custom_input = apply_transforms(custom_input, size=64)

    g_ascent.optimize(conv_layer, 0, input_=custom_input, num_iter=1)

    model.forward.assert_called_with(custom_input)


def test_set_custom_img_size(conv_layer, g_ascent):
    custom_img_size = 64
    g_ascent.img_size = custom_img_size
    assert g_ascent.img_size == custom_img_size

    output = g_ascent.optimize(conv_layer, 0, num_iter=2)

    assert output[0].shape == (1, 3, custom_img_size, custom_img_size)


def test_invalid_layer_str(g_ascent):
    with pytest.raises(TypeError):
        g_ascent.optimize('first', 0, num_iter=2)


def test_invalid_layer_int(g_ascent):
    with pytest.raises(TypeError):
        g_ascent.optimize(0, 0, num_iter=2)


def test_invalid_layer_not_conv(model, g_ascent):
    with pytest.raises(TypeError):
        g_ascent.optimize(model[1], 0, num_iter=2)  # model[1] is  ReLU layer


def test_invalid_filter_idx_not_int(conv_layer, g_ascent):
    with pytest.raises(TypeError):
        g_ascent.optimize(conv_layer, 'first', num_iter=2)


def test_invalid_filter_idx_negative(conv_layer, g_ascent):
    with pytest.raises(ValueError):
        g_ascent.optimize(conv_layer, -1, num_iter=2)


def test_invalid_filter_idx_too_large(conv_layer, g_ascent):
    with pytest.raises(ValueError):
        g_ascent.optimize(conv_layer, 70, num_iter=2)  # the target conv layer has 64 filters


def test_register_forward_hook_to_target_layer(mocker, conv_layer, model):
    mocker.spy(conv_layer, 'register_forward_hook')

    g_ascent = GradientAscent(model)
    g_ascent.optimize(conv_layer, 0, num_iter=2)

    conv_layer.register_forward_hook.assert_called_once()


def test_register_backward_hook_to_first_conv_layer(mocker, conv_layer, model):
    mocker.spy(conv_layer, 'register_backward_hook')

    g_ascent = GradientAscent(model)
    g_ascent.optimize(conv_layer, 0, num_iter=2)

    conv_layer.register_backward_hook.assert_called_once()


def test_remove_any_hooks_before_registering(mocker, conv_layer, model):
    mocker.spy(conv_layer, 'register_forward_hook')
    mocker.spy(conv_layer, 'register_backward_hook')

    another_conv_layer = model[10]
    mocker.spy(another_conv_layer, 'register_forward_hook')
    mocker.spy(another_conv_layer, 'register_backward_hook')

    g_ascent = GradientAscent(model)

    # Optimize for the first conv layer

    g_ascent.optimize(conv_layer, 0, num_iter=2)

    # Optimize for another

    g_ascent.optimize(another_conv_layer, 1, num_iter=2)

    # Backward hook is registered twice, as we always retrieve
    # gradients from it, but forward hook is registered only once

    conv_layer.register_forward_hook.assert_called_once()
    assert conv_layer.register_backward_hook.call_count == 2

    # Instead forward hook is registered on the target layer

    another_conv_layer.register_forward_hook.assert_called_once()


def test_visualize_one_filter(conv_layer, g_ascent):
    output = g_ascent.visualize(conv_layer, 0, 2, return_output=True)

    assert output[-1].shape == (1, 3, g_ascent.img_size, g_ascent.img_size)


def test_visualize_random_filters_from_one_layer(conv_layer, g_ascent):
    num_subplots = 3

    output = g_ascent.visualize(conv_layer, num_iter=2,
                                num_subplots=num_subplots,
                                return_output=True)

    assert len(output) == num_subplots
    assert len(output[0]) == 2  # num_iter
    assert output[0][-1].shape == (1, 3, g_ascent.img_size, g_ascent.img_size)


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


def test_deepdream(conv_layer, g_ascent):
    img_path = path.join(path.dirname(__file__), 'resources', 'test_image.jpg')

    output = g_ascent.deepdream(img_path, conv_layer, 0, return_output=True)

    assert len(output) == 20  # default num_iter for deepdream


if __name__ == '__main__':
    pytest.main([__file__])
