import pytest

from os import path
from PIL import Image

import numpy as np
import torch

from flashtorch.utils import (load_image,
                              apply_transforms,
                              denormalize,
                              standardize_and_clip,
                              format_for_plotting)


#################
# Test fixtures #
#################


@pytest.fixture
def image():
    image_path = path.join(path.dirname(__file__),
                           'resources',
                           'test_image.jpg')

    return load_image(image_path)


##############
# Test cases #
##############


def test_convert_image_to_rgb_when_loading_image(image):
    assert isinstance(image, Image.Image)
    assert image.mode == 'RGB'


def test_handle_non_pil_as_input():
    non_pil_input = np.uint8(np.random.uniform(150, 180, (3, 224, 224)))

    transformed = apply_transforms(non_pil_input)

    assert isinstance(transformed, torch.Tensor)
    assert transformed.requires_grad


def test_handle_non_pil_input_with_channel_last():
    non_pil_input = np.uint8(np.random.uniform(150, 180, (224, 224, 3)))

    transformed = apply_transforms(non_pil_input)

    assert isinstance(transformed, torch.Tensor)
    assert transformed.shape == (1, 3, 224, 224)


def test_transform_image_to_tensor(image):
    transformed = apply_transforms(image)

    assert isinstance(transformed, torch.Tensor)


def test_crop_to_224_by_default(image):
    transformed = apply_transforms(image)

    assert transformed.shape == (1, 3, 224, 224)


def test_crop_to_custom_size(image):
    transformed = apply_transforms(image, 299)

    assert transformed.shape == (1, 3, 299, 299)


def test_denormalize_tensor(image):
    transformed = apply_transforms(image)
    denormalized = denormalize(transformed)

    assert denormalized.shape == transformed.shape
    assert denormalized.min() >= 0.0 and denormalized.max() <= 1.0


def test_format_multi_channel_tensor_with_batch_dimension():
    input_ = torch.zeros((1, 3, 224, 224))

    formatted = format_for_plotting(input_)

    assert formatted.shape == (224, 224, 3)


def test_format_mono_channel_tensor_with_batch_dimension():
    input_ = torch.zeros((1, 1, 224, 224))
    formatted = format_for_plotting(input_)

    assert formatted.shape == (224, 224)


def test_format_multi_channel_tensor_without_batch_dimension():
    input_ = torch.zeros((3, 224, 224))
    formatted = format_for_plotting(input_)

    assert formatted.shape == (224, 224, 3)


def test_format_mono_channel_tensor_without_batch_dimension():
    input_ = torch.zeros((1, 224, 224))
    formatted = format_for_plotting(input_)

    assert formatted.shape == (224, 224)


def test_detach_tensor_from_computational_graph():
    input_ = torch.zeros((1, 224, 224))
    input_.requires_grad = True

    formatted = format_for_plotting(input_)

    assert not formatted.requires_grad


def test_standardize_and_clip_tensor():
    default_min = 0.0
    default_max = 1.0

    input_ = torch.randint(low=-1000, high=1000, size=(224, 224)).float()
    normalized = standardize_and_clip(input_)

    assert normalized.shape == input_.shape
    assert normalized.min().item() >= default_min
    assert normalized.max().item() <= default_max


def test_standardize_and_clip_detach_input_from_graph():
    default_min = 0.0
    default_max = 1.0

    input_ = torch.randint(low=-1000, high=1000, size=(224, 224)).float()
    input_.requires_grad = True
    normalized = standardize_and_clip(input_)

    assert normalized.requires_grad == False


def test_standardize_and_clip_with_custom_min_max():
    custom_min = 2.0
    custom_max = 3.0

    input_ = torch.randint(low=-1000, high=1000, size=(224, 224)).float()
    normalized = standardize_and_clip(input_, min_value=custom_min, max_value=custom_max)

    assert normalized.shape == input_.shape
    assert normalized.min() >= custom_min
    assert normalized.max() <= custom_max


def test_standardize_and_clip_mono_channel_tensor():
    default_min = 0.0
    default_max = 1.0

    input_ = torch.randint(low=-1000, high=1000, size=(1, 224, 224)).float()
    normalized = standardize_and_clip(input_)

    assert normalized.shape == input_.shape
    assert normalized.min().item() >= default_min
    assert normalized.max().item() <= default_max


def test_standardize_and_clip_multi_channel_tensor():
    default_min = 0.0
    default_max = 1.0

    input_ = torch.randint(low=-1000, high=1000, size=(3, 224, 224)).float()
    normalized = standardize_and_clip(input_)

    assert normalized.shape == input_.shape

    for channel in normalized:
        assert channel.min().item() >= default_min
        assert channel.max().item() <= default_max


def test_standardize_and_clip_add_eplison_when_std_is_zero():
    default_min = 0.0
    default_max = 1.0

    input_ = torch.zeros(1, 244, 244)
    normalized = standardize_and_clip(input_)

    assert normalized.shape == input_.shape

    for channel in normalized:
        assert channel.min().item() >= default_min
        assert channel.max().item() <= default_max


if __name__ == '__main__':
    pytest.main([__file__])
