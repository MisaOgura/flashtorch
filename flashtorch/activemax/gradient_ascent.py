import sys

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from flashtorch.utils import (apply_transforms,
                              format_for_plotting,
                              standardize_and_clip)


class GradientAscent(nn.Module):
    """
    """

    def __init__(self, model):
        super().__init__()

        self.model = model
        self.num_layers = len(list(self.model.named_children()))
        self.activation = None
        self.gradients = None

    def _register_forward_hooks(self, layer_idx, filter_idx):
        def _record_activation(module, input_, output):
            self.activation = torch.mean(output[:,filter_idx,:,:])

        self.model[layer_idx].register_forward_hook(_record_activation)

    def _register_backward_hooks(self):
        def _record_gradients(module, grad_in, grad_out):
            if self.gradients.shape == grad_in[0].shape:
                self.gradients = grad_in[0]

        for _, module in self.model.named_modules():
            if isinstance(module, nn.modules.conv.Conv2d) and \
                    module.in_channels == 3:
                module.register_backward_hook(_record_gradients)
                break

    def _ascent_with_adam(self, x, num_iter):
        optimizer = optim.Adam([x], lr=0.01, weight_decay=1e-5)
        optimizer.zero_grad()

        for i in range(num_iter):
            self.model(x)

            self.activation.backward()

            optimizer.step()

        return x

    def _ascent(self, x, num_iter):
        for i in range(num_iter):
            self.model(x)

            self.activation.backward()

            self.gradients /= (torch.sqrt(torch.mean(
                torch.mul(self.gradients, self.gradients))) + 1e-5)

            x = x + self.gradients

        return x

    def _check_indecies(self, layer_idx, filter_idx):
        if not np.issubdtype(type(layer_idx), np.integer) \
                or not np.issubdtype(type(filter_idx), np.integer):
            raise TypeError('Indecies must be integers.')
        elif (layer_idx < 0) or (filter_idx < 0):
            raise ValueError('Indecies must be zero or positive integers.')

        if layer_idx > self.num_layers:
            raise ValueError(f'Layer index must be <= {self.num_layers}.')

        if not isinstance(self.model[layer_idx], nn.modules.conv.Conv2d):
            raise RuntimeError('Layer {layer_idx} is not Conv2d.')

        num_filters = self.model[layer_idx].out_channels

        if filter_idx > num_filters:
            raise ValueError(f'Filter index must be <= {num_filters}.')

    def optimize(self, layer_idx, filter_idx, num_iter, size=224,
                 with_adam=True):
        """
        """

        # Check if the indecies are valid

        self._check_indecies(layer_idx, filter_idx)

        # Register hooks to recort activation and gradients

        self._register_forward_hooks(layer_idx, filter_idx)
        self._register_backward_hooks()

        # Inisialize input noise

        input_noise = np.uint8(np.random.uniform(150, 180, (size, size, 3)))
        input_noise = apply_transforms(input_noise, size=size)

        # Inisialize gradients

        self.gradients = torch.zeros(input_noise.shape)

        # Optimize

        if with_adam:
            output = self._ascent_with_adam(input_noise, num_iter)
        else:
            output = self._ascent(input_noise, num_iter)

        return output

    def visualize_filter(self, layer_idx, filter_idx, num_iter=20, size=224,
                         with_adam=True, figsize=(4, 4), return_output=False):
        """
        """

        output = self.optimize(
            layer_idx, filter_idx, num_iter, size, with_adam)

        plt.figure(figsize=figsize)

        plt.axis('off')
        plt.title(f'Conv2d (layer {layer_idx}, filter {filter_idx})')
        plt.imshow(format_for_plotting(standardize_and_clip(output)));

        # Return output for further processing if desired

        if return_output:
            return output
