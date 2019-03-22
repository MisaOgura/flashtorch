import pdb
import pytest

from os import path
from PIL import Image

import torch

from torchscope.utils import load_image, apply_transforms, denormalize


@pytest.fixture
def image():
    image_path = path.join(path.dirname(__file__), 'test_image.jpg')
    return load_image(image_path)


def test_converts_image_to_rgb(image):
    assert isinstance(image, Image.Image)
    assert image.mode == 'RGB'


def test_transforms_image_to_tensor(image):
    transformed = apply_transforms(image)

    assert isinstance(transformed, torch.Tensor)


def test_crops_to_224(image):
    transformed = apply_transforms(image)

    assert transformed.shape == (1, 3, 224, 224)


def test_denormalize_tensor(image):
    transformed = apply_transforms(image)
    denormalized = denormalize(transformed)

    assert denormalized.shape == (1, 3, 224, 224)
    assert denormalized.min() >= 0.0 and denormalized.max() <= 1.0


if __name__ == '__main__':
    pytest.main([__file__])
