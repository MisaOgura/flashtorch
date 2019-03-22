from PIL import Image

import numpy as np

import torchvision.transforms as transforms


def load_image(image_path):
    """Loads image as a PIL RGB image.

    Args:
        image_path (:obj:`str`): A path to the image.

    Returns:
        An instance of PIL.Image.Image in RGB

    """

    return Image.open(image_path).convert('RGB')


def apply_transforms(image):
    """Transforms a PIL image to torch.Tensor.

    Applies a series of tranformations on PIL image including a conversion
    to a tensor. The returned tensor is ready to be used as an input to the
    neural network.

    First the image is resized to 256, then cropped to 224. The `means` and
    `stds` used for normalization are taken from numbers used in ImageNet, as
    currently developing the package for visualizing pre-trained models.

    The plan is to to expand this to handle custom size/mean/std.

    Args:
        image: An RGB PIL Image

    Shape:
        Input: N/A
        Output: :math:`(N, C, H, W)`

    Returns:
        torch.Tensor

    """
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ])

    return transform(image).unsqueeze(0)


def denormalize(tensor):
    """Reverses the normalization on a tensor.

    Performs a reverse operation on a tensor, so the pixel value range is
    between 0 and 1. Useful for when plotting a tensor into an image.

    Normalization: (image - mean) / std
    Denormalization: image * std + mean

    Args:
        tensor (torch.Tensor): A normalized tensor.

    Shape:
        Input: :math:`(N, C, H, W)`
        Output: :math:`(N, C, H, W)` (same shape as input)

    Return:
        torch.Tensor with pixel values between [0, 1]

    """
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    denormalized = tensor.clone()

    for channel, mean, std in zip(denormalized[0], means, stds):
        channel.mul_(std).add_(mean)

    return denormalized


def save_color_image():
    pass
