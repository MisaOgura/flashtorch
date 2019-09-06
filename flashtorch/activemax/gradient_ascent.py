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

    def __init__(self, model, img_size=128, lr=0.01, weight_decay=1e-5,
                 with_adam=True):
        super().__init__()

        self.model = model
        self._img_size = img_size
        self._lr = lr
        self._weight_decay = weight_decay
        self._with_adam = with_adam

        self.num_layers = len(list(self.model.named_children()))
        self.activation = None
        self.gradients = None
        self.output = None

    @property
    def lr(self):
        return self._lr

    @lr.setter
    def lr(self, lr):
        self._lr = lr

    @property
    def weight_decay(self):
        return self._weight_decay

    @weight_decay.setter
    def weight_decay(self, weight_decay):
        self._weight_decay = weight_decay

    @property
    def img_size(self):
        return self._img_size

    @img_size.setter
    def img_size(self, img_size):
        self._img_size = img_size

    @property
    def with_adam(self):
        return self._with_adam

    @with_adam.setter
    def with_adam(self, with_adam):
        self._with_adam = with_adam

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
        optimizer = optim.Adam([x],
                               lr=self._lr,
                               weight_decay=self.weight_decay)

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

    def _validate_layer_idx(self, layer_idx):
        if not np.issubdtype(type(layer_idx), np.integer):
            raise TypeError('Indecies must be integers.')
        elif (layer_idx < 0) or (layer_idx > self.num_layers):
            raise ValueError(f'Layer index must be between 0 and {self.num_layers - 1}.')

        if not isinstance(self.model[layer_idx], nn.modules.conv.Conv2d):
            raise RuntimeError('Layer {layer_idx} is not Conv2d.')

    def _validate_filter_idx(self, num_filters, filter_idx):
        if not np.issubdtype(type(filter_idx), np.integer):
            raise TypeError('Indecies must be integers.')
        elif (filter_idx < 0) or (filter_idx > num_filters):
            raise ValueError(f'Filter index must be between 0 and {num_filters - 1}.')

    def _validate_indicies(self, layer_idx, filter_idxs):
        self._validate_layer_idx(layer_idx)
        num_filters = self.model[layer_idx].out_channels

        for filter_idx in filter_idxs:
            self._validate_filter_idx(num_filters, filter_idx)

    def _visualize_filter(self, layer_idx, filter_idx, num_iter, figsize):
        """
        """

        self._validate_indicies(layer_idx, [filter_idx])

        self.output = self.optimize(layer_idx, filter_idx, num_iter)

        plt.figure(figsize=figsize)
        plt.axis('off')
        plt.title(f'Conv2d (layer {layer_idx}, filter {filter_idx})')

        plt.imshow(format_for_plotting(standardize_and_clip(self.output)));

    def _visualize_filters(self, layer_idx, filter_idxs, num_iter, num_subplots):
        """
        """

        # Prepare the main plot

        num_cols = 4
        num_rows = int(np.ceil(num_subplots / num_cols))

        fig = plt.figure(figsize=(16, num_rows * 5))
        plt.title(f'Conv2d layer {layer_idx}')
        plt.axis('off')

        self.output = []

        # Plot subplots

        for i, filter_idx in enumerate(filter_idxs):
            output = self.optimize(layer_idx, filter_idx, num_iter)

            self.output.append(output)

            ax = fig.add_subplot(num_rows, num_cols, i+1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'filter {filter_idx}')
            ax.imshow(format_for_plotting(standardize_and_clip(output)))

        plt.subplots_adjust(wspace=0, hspace=0);

    def optimize(self, layer_idx, filter_idx, num_iter):
        """
        """

        # Check if the indecies are valid

        self._validate_indicies(layer_idx, [filter_idx])

        # Register hooks to recort activation and gradients

        self._register_forward_hooks(layer_idx, filter_idx)
        self._register_backward_hooks()

        # Inisialize input noise

        input_noise = np.uint8(np.random.uniform(
            150, 180, (self._img_size, self._img_size, 3)))
        input_noise = apply_transforms(input_noise, size=self._img_size)

        # Inisialize gradients

        self.gradients = torch.zeros(input_noise.shape)

        # Optimize

        if self.with_adam:
            output = self._ascent_with_adam(input_noise, num_iter)
        else:
            output = self._ascent(input_noise, num_iter)

        return output

    def visualize(self, layer_idx, filter_idxs=None, num_iter=20,
                  num_subplots=5, return_output=False):

        if (type(filter_idxs) == int):
            output = self._visualize_filter(layer_idx,
                                            filter_idxs,
                                            num_iter=num_iter,
                                            figsize=(4, 4))
        else:
            if filter_idxs is None:
                num_total_filters = self.model[layer_idx].out_channels
                num_subplots = min(num_total_filters, num_subplots)

                filter_idxs = np.random.choice(range(num_total_filters),
                                               size=num_subplots)
            else:
                self._validate_indicies(layer_idx, filter_idxs)
                num_subplots = len(filter_idxs)

            self._visualize_filters(layer_idx,
                                    filter_idxs,
                                    num_iter,
                                    num_subplots)

        if return_output:
            return self.output
