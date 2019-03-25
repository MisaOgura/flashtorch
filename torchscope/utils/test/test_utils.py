import pdb
import pytest

from os import path
from PIL import Image

import torch

from torchscope.utils import (load_image,
                              apply_transforms,
                              denormalize,
                              normalize,
                              format_for_plotting)


@pytest.fixture
def image():
    image_path = path.join(path.dirname(__file__), 'test_image.jpg')
    return load_image(image_path)


def test_convert_image_to_rgb(image):
    assert isinstance(image, Image.Image)
    assert image.mode == 'RGB'


def test_transform_image_to_tensor(image):
    transformed = apply_transforms(image)

    assert isinstance(transformed, torch.Tensor)


def test_crop_to_224(image):
    transformed = apply_transforms(image)

    assert transformed.shape == (1, 3, 224, 224)


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


# def test_normalize(image):
#     normalized = normalize(image)

#     assert normalized.min() >= 0.0 and normalized.max() <= 1.0


if __name__ == '__main__':
    pytest.main([__file__])
