import inspect
import pdb
import pytest

from sys import stdout

import scipy
import torch
import torchvision
import torchvision.models as models

from torchscope.saliency import Backprop


@pytest.fixture
def model():
    return models.alexnet()


def test_set_model_to_eval_mode(mocker, model):

    mocker.spy(model, 'eval')
    Backprop(model)

    model.eval.assert_called_once()


def test_find_first_conv_layer_in_torchvision_models(mocker):
    available_models = inspect.getmembers(models, inspect.isfunction)

    print()
    for name, func in available_models:
        print(f'Finding the first conv layer in model: {name}', end='\r')
        stdout.write('\x1b[2K')

        model = func()
        mocker.spy(model, 'eval')

        backprop = Backprop(model)
        target_layer = backprop._find_target_layer()

        assert type(target_layer) == torch.nn.modules.conv.Conv2d
        assert target_layer.in_channels == 3


def test_register_hook(mocker, model):
    target_layer = model.features[0]
    mocker.spy(target_layer, 'register_backward_hook')

    Backprop(model)

    target_layer.register_backward_hook.assert_called_once()


def test_zero_out_gradient(mocker, model):
    backprop = Backprop(model)
    mocker.spy(model, 'zero_grad')

    target_class = 1
    input_ = torch.zeros([1, 3, 224, 224])

    backprop.calculate_gradient(input_, target_class)

    model.zero_grad.assert_called_once()


def test_calculate_gradient_of_target_class_only(mocker, model):
    backprop = Backprop(model)

    # Mock the output from the neural network

    num_classes = 10
    mock_tensor = torch.zeros((1, num_classes))
    mock_output = mocker.Mock(spec=mock_tensor, shape=(1, num_classes))
    mocker.patch.object(model, 'forward', return_value=mock_output)

    target_class = 5
    input_ = torch.zeros([1, 3, 224, 224])

    backprop.calculate_gradient(input_, target_class)

    expected_gradient_target = torch.zeros((1, num_classes))
    expected_gradient_target[0][target_class] = 1

    args, kwargs = mock_output.backward.call_args

    assert torch.all(kwargs['gradient'].eq(expected_gradient_target))


def test_calculate_gradient_wrt_inputs(mocker, model):
    backprop = Backprop(model)

    target_class = 1
    input_ = torch.zeros([1, 3, 224, 224])

    gradient = backprop.calculate_gradient(input_, target_class)

    assert gradient.shape == (1, 224, 224)
