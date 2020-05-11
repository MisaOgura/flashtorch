import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from flashtorch.utils import (apply_transforms,
                              format_for_plotting,
                              load_image,
                              standardize_and_clip)


class GradientAscent:
    """Provides an interface for activation maximization via gradient descent.

    This class implements the gradient ascent algorithm in order to perform
    activation maximization with convolutional neural networks (CNN).

    `Activation maximization <https://pdfs.semanticscholar.org/65d9/94fb778a8d9e0f632659fb33a082949a50d3.pdf>`_
    is one form of feature visualization that allows us to visualize what CNN
    filters are "looking for", by applying each filter to an input image and
    updating the input image so as to maximize the activation of the filter of
    interest (i.e. treating it as a gradient ascent task with activation as the
    loss). The implementation is inspired by `this demo <https://blog.keras.io/category/demo.html>`_
    by Francois Chollet.

    Args:
        model: A neural network model from `torchvision.models
            <https://pytorch.org/docs/stable/torchvision/models.html>`_,
            typically without the fully-connected part of the network.
            e.g. torchvisions.alexnet(pretrained=True).features
        img_size (int, optional, default=224): The size of an input image to be
            optimized.
        lr (float, optional, default=1.): The step size (or learning rate) of
            the gradient ascent.
        use_gpu (bool, optional, default=False): Use GPU if set to True and
            `torch.cuda.is_available()`.

    """ # noqa

    ####################
    # Public interface #
    ####################

    def __init__(self, model, img_size=224, lr=1., use_gpu=False):
        self.model = model
        self._img_size = img_size
        self._lr = lr
        self._use_gpu = use_gpu

        self.num_layers = len(list(self.model.named_children()))
        self.activation = None
        self.gradients = None

        self.handlers = []

        self.output = None

    @property
    def lr(self):
        return self._lr

    @lr.setter
    def lr(self, lr):
        self._lr = lr

    @property
    def img_size(self):
        return self._img_size

    @img_size.setter
    def img_size(self, img_size):
        self._img_size = img_size

    @property
    def use_gpu(self):
        return self._use_gpu

    @use_gpu.setter
    def use_gpu(self, use_gpu):
        self._use_gpu = use_gpu

    def optimize(self, layer, filter_idx, input_=None, num_iter=30):
        """Generates an image that maximally activates the target filter.

        Args:
            layer (torch.nn.modules.conv.Conv2d): The target Conv2d layer from
                which the filter to be chosen, based on `filter_idx`.
            filter_idx (int): The index of the target filter.
            num_iter (int, optional, default=30): The number of iteration for
                the gradient ascent operation.

        Returns:
            output (list of torch.Tensor): With dimentions
                :math:`(num_iter, C, H, W)`. The size of the image is
                determined by `img_size` attribute which defaults to 224.

        """

        # Validate the type of the layer

        if type(layer) != nn.modules.conv.Conv2d:
            raise TypeError('The layer must be nn.modules.conv.Conv2d.')

        # Validate filter index

        num_total_filters = layer.out_channels
        self._validate_filter_idx(num_total_filters, filter_idx)

        # Inisialize input (as noise) if not provided

        if input_ is None:
            input_ = np.uint8(np.random.uniform(
                150, 180, (self._img_size, self._img_size, 3)))
            input_ = apply_transforms(input_, size=self._img_size)

        if torch.cuda.is_available() and self.use_gpu:
            self.model = self.model.to('cuda')
            input_ = input_.to('cuda')

        # Remove previous hooks if any

        while len(self.handlers) > 0:
            self.handlers.pop().remove()

        # Register hooks to record activation and gradients

        self.handlers.append(self._register_forward_hooks(layer, filter_idx))
        self.handlers.append(self._register_backward_hooks())

        # Inisialize gradients

        self.gradients = torch.zeros(input_.shape)

        # Optimize

        return self._ascent(input_, num_iter)

    def visualize(self, layer, filter_idxs=None, lr=1., num_iter=30,
                  num_subplots=4, figsize=(4, 4), title='Conv2d',
                  return_output=False):
        """Optimizes for the target layer/filter and visualizes the output.

        A method that combines optimization and visualization. There are
        mainly 3 types of operations, given a target layer:

        1. If `filter_idxs` is provided as an integer, it optimizes for the
            filter specified and plots the output.
        2. If `filter_idxs` is provided as a list of integers, it optimizes for
            all the filters specified and plots the output.
        3. if `filter_idx` is not provided, i.e. None, it randomly chooses
            `num_subplots` number of filters from the layer provided and
            plots the output.

        It also returns the output of the optimization, if specified with
        `return_output=True`.

        Args:
            layer (torch.nn.modules.conv.Conv2d): The target Conv2d layer from
                which the filter to be chosen, based on `filter_idx`.
            filter_idxs (int or list of int, optional, default=None): The index
                or indecies of the target filter(s).
            lr (float, optional, default=.1): The step size of optimization.
            num_iter (int, optional, default=30): The number of iteration for
                the gradient ascent operation.
            num_subplots (int, optional, default=4): The number of filters to
                optimize for and visualize. Relevant in case 3 above.
            figsize (tuple, optional, default=(4, 4)): The size of the plot.
                Relevant in case 1 above.
            title (str, optional default='Conv2d'): The title of the plot.
            return_output (bool, optional, default=False): Returns the
                output(s) of optimization if set to True.


        Returns:
            For a single optimization (i.e. case 1 above):
                output (list of torch.Tensor): With dimentions
                    :math:`(num_iter, C, H, W)`. The size of the image is
                    determined by `img_size` attribute which defaults to 224.
            For multiple optimization (i.e. case 2 or 3 above):
                output (list of list of torch.Tensor): With dimentions
                    :math:`(num_subplots, num_iter, C, H, W)`. The size of the
                    image is determined by `img_size` attribute which defaults
                    to 224.

        """

        self._lr = lr

        if (type(filter_idxs) == int):
            self._visualize_filter(layer,
                                   filter_idxs,
                                   num_iter=num_iter,
                                   figsize=figsize,
                                   title=title)
        else:
            num_total_filters = layer.out_channels

            if filter_idxs is None:
                num_subplots = min(num_total_filters, num_subplots)
                filter_idxs = np.random.choice(range(num_total_filters),
                                               size=num_subplots)

            self._visualize_filters(layer,
                                    filter_idxs,
                                    num_iter,
                                    len(filter_idxs),
                                    title=title)

        if return_output:
            return self.output

    def deepdream(self, img_path, layer, filter_idx, lr=.1, num_iter=20,
                  figsize=(4, 4), title='DeepDream', return_output=False):
        """Creates DeepDream.

        It applies the optimization on the image provided. The image is loaded
        and made into a torch.Tensor that is compatible as the input to the
        network.

        Read the original blog post by Google for more information on
        `DeepDream <https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html>`_.

        Args:
            img_path (str): A path to the image you want to apply DeepDream on
            layer (torch.nn.modules.conv.Conv2d): The target Conv2d layer from
                which the filter to be chosen, based on `filter_idx`.
            filter_idx (int): The index of the target filter.
            lr (float, optional, default=.1): The step size of optimization.
            num_iter (int, optional, default=30): The number of iteration for
                the gradient ascent operation.
            figsize (tuple, optional, default=(4, 4)): The size of the plot.
                Relevant in case 1 above.
            title (str, optional default='Conv2d'): The title of the plot.
            return_output (bool, optional, default=False): Returns the
                output(s) of optimization if set to True.

        Returns:
            output (list of torch.Tensor): With dimentions
                :math:`(num_iter, C, H, W)`. The size of the image is
                determined by `img_size` attribute which defaults to 224.

        """ # noqa

        input_ = apply_transforms(load_image(img_path), self.img_size)

        self._lr = lr
        output = self.optimize(layer, filter_idx, input_, num_iter=num_iter)

        plt.figure(figsize=figsize)
        plt.axis('off')
        plt.title(title)

        plt.imshow(format_for_plotting(
            standardize_and_clip(output[-1],
                                 saturation=0.15,
                                 brightness=0.7))); # noqa

        if return_output:
            return output

    #####################
    # Private interface #
    #####################

    def _register_forward_hooks(self, layer, filter_idx):
        def _record_activation(module, input_, output):
            self.activation = torch.mean(output[:, filter_idx, :, :])

        return layer.register_forward_hook(_record_activation)

    def _register_backward_hooks(self):
        def _record_gradients(module, grad_in, grad_out):
            if self.gradients.shape == grad_in[0].shape:
                self.gradients = grad_in[0]

        for _, module in self.model.named_modules():
            if isinstance(module, nn.modules.conv.Conv2d) and \
                    module.in_channels == 3:
                return module.register_backward_hook(_record_gradients)

    def _ascent(self, x, num_iter):
        output = []

        for i in range(num_iter):
            self.model(x)

            self.activation.backward()

            self.gradients /= (torch.sqrt(torch.mean(
                torch.mul(self.gradients, self.gradients))) + 1e-5)

            x = x + self.gradients * self._lr
            output.append(x)

        return output

    def _validate_filter_idx(self, num_filters, filter_idx):
        if not np.issubdtype(type(filter_idx), np.integer):
            raise TypeError('Indecies must be integers.')
        elif (filter_idx < 0) or (filter_idx > num_filters):
            raise ValueError(f'Filter index must be between 0 and \
                             {num_filters - 1}.')

    def _visualize_filter(self, layer, filter_idx, num_iter, figsize, title):
        self.output = self.optimize(layer, filter_idx, num_iter=num_iter)

        plt.figure(figsize=figsize)
        plt.axis('off')
        plt.title(title)

        plt.imshow(format_for_plotting(
            standardize_and_clip(self.output[-1],
                                 saturation=0.15,
                                 brightness=0.7))); # noqa

    def _visualize_filters(self, layer, filter_idxs, num_iter, num_subplots,
                           title):
        # Prepare the main plot

        num_cols = 4
        num_rows = int(np.ceil(num_subplots / num_cols))

        fig = plt.figure(figsize=(16, num_rows * 5))
        plt.title(title)
        plt.axis('off')

        self.output = []

        # Plot subplots

        for i, filter_idx in enumerate(filter_idxs):
            output = self.optimize(layer, filter_idx, num_iter=num_iter)

            self.output.append(output)

            ax = fig.add_subplot(num_rows, num_cols, i+1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'filter {filter_idx}')

            ax.imshow(format_for_plotting(
                standardize_and_clip(output[-1],
                                     saturation=0.15,
                                     brightness=0.7)))

        plt.subplots_adjust(wspace=0, hspace=0); # noqa
