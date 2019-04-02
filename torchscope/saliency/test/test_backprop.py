import inspect
import pytest

from sys import stdout

import scipy

import torch
import torch.nn as nn

import torchvision
import torchvision.models as models

from torchscope.saliency import Backprop


def find_target_layer(model, layer_type, in_channels):
    for _, module in model.named_modules():
        if isinstance(module, layer_type) and \
                module.in_channels == in_channels:
            return module


@pytest.fixture
def model():
    return models.alexnet()


def test_set_model_to_eval_mode(mocker, model):

    mocker.spy(model, 'eval')
    Backprop(model)

    model.eval.assert_called_once()


def test_register_backward_hook_to_the_right_layer(mocker):
    available_models = inspect.getmembers(models, inspect.isfunction)

    print()
    for name, func in available_models:
        print(f'Finding the first conv layer in model: {name}', end='\r')
        stdout.write('\x1b[2K')

        model = func()
        mocker.spy(model, 'eval')

        backprop = Backprop(model)
        target_layer = find_target_layer(model,
                                         nn.modules.conv.Conv2d,
                                         3)

        mocker.spy(target_layer, 'register_backward_hook')

        Backprop(model)

        target_layer.register_backward_hook.assert_called_once()


def test_zero_out_gradients(mocker, model):
    backprop = Backprop(model)
    mocker.spy(model, 'zero_grad')

    target_class = 1
    input_ = torch.zeros([1, 3, 224, 224])

    backprop.calculate_gradients(input_, target_class)

    model.zero_grad.assert_called_once()


def test_calculate_gradients_of_target_class_only(mocker, model):
    backprop = Backprop(model)

    # Mock the output from the neural network

    num_classes = 10
    mock_tensor = torch.zeros((1, num_classes))
    mock_output = mocker.Mock(spec=mock_tensor, shape=(1, num_classes))
    mocker.patch.object(model, 'forward', return_value=mock_output)

    target_class = 5
    input_ = torch.zeros([1, 3, 224, 224])

    backprop.calculate_gradients(input_, target_class)

    expected_gradients_target = torch.zeros((1, num_classes))
    expected_gradients_target[0][target_class] = 1

    args, kwargs = mock_output.backward.call_args

    assert torch.all(kwargs['gradient'].eq(expected_gradients_target))


def test_calculate_gradients_wrt_inputs(mocker, model):
    backprop = Backprop(model)

    target_class = 1
    input_ = torch.zeros([1, 3, 224, 224])

    gradients = backprop.calculate_gradients(input_, target_class)

    assert gradients.shape == (3, 224, 224)


def test_return_max_across_color_channel_if_specified(mocker, model):
    backprop = Backprop(model)

    target_class = 1
    input_ = torch.zeros([1, 3, 224, 224])

    gradients = backprop.calculate_gradients(input_, target_class, take_max=True)

    assert gradients.shape == (1, 224, 224)
