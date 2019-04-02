#!/usr/bin/env python
"""
"""

import torch
import torch.nn as nn


class Backprop:
    """Provides an interface to perform backpropagation.

    This class provids a way to calculate a gradients of a target class output
    w.r.t. an input image, by performing a single backprobagation.

    The gradients obtained can be used to visualise an image-specific class
    saliency map, which can gives some intuition on regions within the input
    image that contribute the most (and least) to the corresponding output.

    More details on saliency maps: `Deep Inside Convolutional Networks:
    Visualising Image Classification Models and Saliency Maps
    <https://arxiv.org/pdf/1312.6034.pdf>`_.

    Args:
        model: A neural network model from `torchvision.models
            <https://pytorch.org/docs/stable/torchvision/models.html>`_.
        device (str, optional): 'cpu' or 'cuda'. Defaults to 'cpu'.

    """

    def __init__(self, model):
        self.model = model
        self.model.eval()

        self.gradients = None

        self._register_hooks()
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def calculate_gradients(self, input_, target_class, take_max=False):
        """Calculates gradients of the target_class output w.r.t. an input_.

        The gradients is calculated for each colour channel. Then, the maximum
        gradients across colour channels is returned.

        Args:
            input_ (torch.Tensor): With shape :math:`(N, C, H, W)`.
            target_class (int)

        Returns:
            gradients (torch.Tensor): With shape :math:`(C, H, W)`.

        """

        self.model = self.model.to(self._device)
        self.model.zero_grad()

        input_ = input_.to(self._device)
        input_.requires_grad = True

        self.gradients = torch.zeros(input_.shape)

        # Get a raw prediction value (logit) from the last linear layer

        output = self.model(input_)

        # Create a 2D tensor with shape (1, num_classes) and
        # set all element to zero

        target = torch.FloatTensor(1, output.shape[-1]).zero_()

        # Set the element at target class index to be 1

        target[0][target_class] = 1

        # Calculate gradients of the target class output w.r.t. input_

        output.backward(gradient=target)

        # Detach the gradients from the graph and move to cpu

        gradients = self.gradients.detach().cpu()[0]

        if take_max:
            # Take the maximum across colour channels

            gradients = gradients.max(dim=0, keepdim=True)[0]

        return gradients

    def _register_hooks(self):
        def _record_gradients(module, grad_in, grad_out):
            if self.gradients.shape == grad_in[0].shape:
                self.gradients = grad_in[0]

        for _, module in self.model.named_modules():
            if isinstance(module, nn.modules.conv.Conv2d) and \
                    module.in_channels == 3:
                module.register_backward_hook(_record_gradients)
