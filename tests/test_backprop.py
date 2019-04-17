import inspect
import pytest

from sys import stdout

import scipy

import torch
import torch.nn as nn

import torchvision
import torchvision.models as models

from torchscope.saliency import Backprop


#####################
# Utility functions #
#####################


def find_first_conv_layer(model, layer_type, in_channels):
    for _, module in model.named_modules():
        if isinstance(module, layer_type) and \
                module.in_channels == in_channels:
            return module


def find_relu_layers(model, layer_type):
    modules = []

    for _, module in model.named_modules():
        if isinstance(module, layer_type):
            modules.append(module)

    return modules


def make_mock_output(mocker, model, num_classes, top_class):
    # Mock the output from the neural network

    mock_tensor = torch.zeros((1, num_classes))
    mock_tensor[0][top_class] = 1
    mock_output = mocker.Mock(spec=mock_tensor, shape=(1, num_classes))

    mocker.patch.object(model, 'forward', return_value=mock_output)

    # Mock the return value of output.topk()

    mock_topk = (None, top_class)
    mocker.patch.object(mock_output, 'topk', return_value=mock_topk)

    return mock_output


#################
# Test fixtures #
#################


@pytest.fixture
def model():
    return models.alexnet()


@pytest.fixture
def available_models():
    return inspect.getmembers(models, inspect.isfunction)


##############
# Test cases #
##############


def test_set_model_to_eval_mode(mocker, model):
    mocker.spy(model, 'eval')
    Backprop(model)

    model.eval.assert_called_once()


def test_zero_out_gradients(mocker, model):
    backprop = Backprop(model)
    mocker.spy(model, 'zero_grad')

    num_classes = 10
    target_class = 5
    input_ = torch.zeros([1, 3, 224, 224])

    make_mock_output(mocker, model, num_classes, target_class)

    backprop.calculate_gradients(input_, target_class)

    model.zero_grad.assert_called_once()


def test_calculate_gradients_of_target_class_only(mocker, model):
    backprop = Backprop(model)

    num_classes = 10
    target_class = 5
    input_ = torch.zeros([1, 3, 224, 224])

    # Mock the output from the neural network

    mock_output = make_mock_output(mocker, model, num_classes, target_class)

    backprop.calculate_gradients(input_, target_class)

    # Make expected target of the gradient calculation

    num_classes = 10

    expected_gradients_target = torch.zeros((1, num_classes))
    expected_gradients_target[0][target_class] = 1

    args, kwargs = mock_output.backward.call_args

    assert torch.all(kwargs['gradient'].eq(expected_gradients_target))


def test_return_max_across_color_channels_if_specified(mocker, model):
    backprop = Backprop(model)

    num_classes = 10
    target_class = 5
    input_ = torch.zeros([1, 3, 224, 224])

    make_mock_output(mocker, model, num_classes, target_class)

    gradients = backprop.calculate_gradients(input_,
                                             target_class,
                                             take_max=True)

    assert gradients.shape == (1, 224, 224)


def test_checks_input_size_for_inception_model(mocker):
    with pytest.raises(ValueError) as error:
        model = models.inception_v3()
        backprop = Backprop(model)

        target_class = 5
        input_ = torch.zeros([1, 3, 224, 224])

        backprop.calculate_gradients(input_, target_class)

    assert 'Image must be 299x299 for Inception models.' in str(error.value)


def test_raise_when_prediction_is_wrong(mocker, model):
    with pytest.raises(ValueError) as error:
        backprop = Backprop(model)

        num_classes = 10
        target_class = 5

        input_ = torch.zeros([1, 3, 224, 224])

        # Mock a wrong prediction

        predict_class = 1
        make_mock_output(mocker, model, num_classes, predict_class)

        backprop.calculate_gradients(input_, target_class)

    assert 'The network prediction was wrong' in str(error.value)


# Test compatibilities with torchvision models


def test_register_backward_hook_to_first_conv_layer(mocker, available_models):
    print()
    for name, model in available_models:
        print(f'Testing model: {name}', end='\r')
        stdout.write('\x1b[2K')

        model = model()

        conv_layer = find_first_conv_layer(model, nn.modules.conv.Conv2d, 3)
        mocker.spy(conv_layer, 'register_backward_hook')

        Backprop(model)

        conv_layer.register_backward_hook.assert_called_once()


def test_register_hooks_to_relu_layers(mocker, available_models):
    print()
    for name, model in available_models:
        print(f'Testing model: {name}', end='\r')
        stdout.write('\x1b[2K')

        model = model()

        relu_layers = find_relu_layers(model,nn.ReLU)

        for layer in relu_layers:
            mocker.spy(layer, 'register_forward_hook')
            mocker.spy(layer, 'register_backward_hook')

            Backprop(model, guided=True)

            layer.register_forward_hook.assert_called_once()
            layer.register_backward_hook.assert_called_once()


def test_calculate_gradients_for_all_models(mocker, available_models):
    print()
    for name, model in available_models:
        print(f'Testing model: {name}', end='\r')
        stdout.write('\x1b[2K')

        model = model()
        backprop = Backprop(model)

        num_classes = 10
        target_class = 5
        input_ = torch.zeros([1, 3, 224, 224])

        if 'inception' in name:
            input_ = torch.zeros([1, 3, 299, 299])

        make_mock_output(mocker, model, num_classes, target_class)

        gradients = backprop.calculate_gradients(input_, target_class)

        assert gradients.shape == input_.size()[1:]


if __name__ == '__main__':
    pytest.main([__file__])
