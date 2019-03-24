
"""torchscope.utils

This module provides utility functions for image handling and tensor
transformation.

"""

from PIL import Image

import torchvision.transforms as transforms


def load_image(image_path):
    """Loads image as a PIL RGB image.

    Args:
        image_path (str): A path to the image.

    Returns:
        An instance of PIL.Image.Image in RGB

    """

    return Image.open(image_path).convert('RGB')


def apply_transforms(image):
    """Transforms a PIL image to torch.Tensor.

    Applies a series of tranformations on PIL image including a conversion
    to a tensor. The returned tensor has a shape of :math:`(N, C, H, W)` and
    is ready to be used as an input to neural networks.

    First the image is resized to 256, then cropped to 224. The `means` and
    `stds` for normalisation are taken from numbers used in ImageNet, as
    currently developing the package for visualizing pre-trained models.

    The plan is to to expand this to handle custom size/mean/std.

    Args:
        image (PIL.Image.Image): RGB PIL Image

    Shape:
        Input: N/A
        Output: :math:`(N, C, H, W)`

    Returns:
        torch.Tensor (torch.float32): Transformed image tensor.

    Note:
        Symbols used to describe dimensions:
            - N: number of images in the batch
            - C: number of channels
            - H: height of the image
            - W: width of the image

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
    """Reverses the normalisation on a tensor.

    Performs a reverse operation on a tensor, so the pixel value range is
    between 0 and 1. Useful for when plotting a tensor into an image.

    Normalisation: (image - mean) / std
    Denormalisation: image * std + mean

    Args:
        tensor (torch.Tensor, dtype=torch.float32): Normalized image tensor.

    Shape:
        Input: :math:`(N, C, H, W)`
        Output: :math:`(N, C, H, W)` (same shape as input)

    Return:
        torch.Tensor (torch.float32): Demornalied image tensor with pixel
            values between [0, 1].

    Note:
        Symbols used to describe dimensions:
            - N: number of images in the batch
            - C: number of channels
            - H: height of the image
            - W: width of the image

    """
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    denormalized = tensor.clone()

    for channel, mean, std in zip(denormalized[0], means, stds):
        channel.mul_(std).add_(mean)

    return denormalized


def format_for_plotting(tensor):
    """Formats the shape of tensor for plotting.

    Tensors typically have a shape of :math:`(N, C, H, W)`, which is not
    suitable for plotting as images. This function formats their shape into
    :math:`(H, W, C)` by removing the batch dimension and pushing the channel
    dimention to the last.

    Args:
        tensor (torch.Tensor, torch.float32): Image tensor.

    Shape:
        Input: :math:`(N, C, H, W)`
        Output: :math:`(H, W, C)`

    Return:
        torch.Tensor (torch.float32): Formatted image tensor.

    Note:
        Symbols used to describe dimensions:
            - N: number of images in the batch
            - C: number of channels
            - H: height of the image
            - W: width of the image

    """
    return tensor.squeeze(0).permute(1, 2, 0)
