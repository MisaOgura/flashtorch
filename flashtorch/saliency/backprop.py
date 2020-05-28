import warnings

import matplotlib.pyplot as plt

import torch
import torch.nn as nn


from flashtorch.utils import (denormalize,
                              format_for_plotting,
                              standardize_and_clip)


class Backprop:
    """Provides an interface to perform backpropagation.

    This class provids a way to calculate the gradients of a target class
    output w.r.t. an input image, by performing a single backprobagation.

    The gradients obtained can be used to visualise an image-specific class
    saliency map, which can gives some intuition on regions within the input
    image that contribute the most (and least) to the corresponding output.

    More details on saliency maps: `Deep Inside Convolutional Networks:
    Visualising Image Classification Models and Saliency Maps
    <https://arxiv.org/pdf/1312.6034.pdf>`_.

    Args:
        model: A neural network model from `torchvision.models
            <https://pytorch.org/docs/stable/torchvision/models.html>`_.

    """ # noqa

    ####################
    # Public interface #
    ####################

    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.gradients = None
        self._register_conv_hook()

    def calculate_gradients(self,
                            input_,
                            target_class=None,
                            take_max=False,
                            guided=False,
                            use_gpu=False):

        """Calculates gradients of the target_class output w.r.t. an input_.

        The gradients is calculated for each colour channel. Then, the maximum
        gradients across colour channels is returned.

        Args:
            input_ (torch.Tensor): With shape :math:`(N, C, H, W)`.
            target_class (int, optional, default=None)
            take_max (bool, optional, default=False): If True, take the maximum
                gradients across colour channels for each pixel.
            guided (bool, optional, default=Fakse): If True, perform guided
                backpropagation. See `Striving for Simplicity: The All
                Convolutional Net <https://arxiv.org/pdf/1412.6806.pdf>`_.
            use_gpu (bool, optional, default=False): Use GPU if set to True and
                `torch.cuda.is_available()`.

        Returns:
            gradients (torch.Tensor): With shape :math:`(C, H, W)`.

        """ # noqa

        if 'inception' in self.model.__class__.__name__.lower():
            if input_.size()[1:] != (3, 299, 299):
                raise ValueError('Image must be 299x299 for Inception models.')

        if guided:
            self.relu_outputs = []
            self._register_relu_hooks()

        if torch.cuda.is_available() and use_gpu:
            self.model = self.model.to('cuda')
            input_ = input_.to('cuda')

        self.model.zero_grad()

        self.gradients = torch.zeros(input_.shape)

        # Get a raw prediction value (logit) from the last linear layer

        output = self.model(input_)

        # Don't set the gradient target if the model is a binary classifier
        # i.e. has one class prediction

        if len(output.shape) == 1:
            target = None
        else:
            _, top_class = output.topk(1, dim=1)

            # Create a 2D tensor with shape (1, num_classes) and
            # set all element to zero

            target = torch.FloatTensor(1, output.shape[-1]).zero_()

            if torch.cuda.is_available() and use_gpu:
                target = target.to('cuda')

            if (target_class is not None) and (top_class != target_class):
                warnings.warn(UserWarning(
                    f'The predicted class index {top_class.item()} does not' +
                    f'equal the target class index {target_class}. ' +
                    'Calculating the gradient w.r.t. the predicted class.'
                ))

            # Set the element at top class index to be 1

            target[0][top_class] = 1

        # Calculate gradients of the target class output w.r.t. input_

        output.backward(gradient=target)

        # Detach the gradients from the graph and move to cpu

        gradients = self.gradients.detach().cpu()[0]

        if take_max:
            # Take the maximum across colour channels

            gradients = gradients.max(dim=0, keepdim=True)[0]

        return gradients

    def visualize(self, input_, target_class, guided=False, use_gpu=False,
                  figsize=(16, 4), cmap='viridis', alpha=.5,
                  return_output=False):
        """Calculates gradients and visualizes the output.

        A method that combines the backprop operation and visualization.

        It also returns the gradients, if specified with `return_output=True`.

        Args:
            input_ (torch.Tensor): With shape :math:`(N, C, H, W)`.
            target_class (int, optional, default=None)
            take_max (bool, optional, default=False): If True, take the maximum
                gradients across colour channels for each pixel.
            guided (bool, optional, default=Fakse): If True, perform guided
                backpropagation. See `Striving for Simplicity: The All
                Convolutional Net <https://arxiv.org/pdf/1412.6806.pdf>`_.
            use_gpu (bool, optional, default=False): Use GPU if set to True and
                `torch.cuda.is_available()`.
            figsize (tuple, optional, default=(16, 4)): The size of the plot.
            cmap (str, optional, default='viridis): The color map of the
                gradients plots. See avaialable color maps `here <https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html>`_.
            alpha (float, optional, default=.5): The alpha value of the max
                gradients to be jaxaposed on top of the input image.
            return_output (bool, optional, default=False): Returns the
                output(s) of optimization if set to True.

        Returns:
            gradients (torch.Tensor): With shape :math:`(C, H, W)`.
        """ # noqa

        # Calculate gradients

        gradients = self.calculate_gradients(input_,
                                             target_class,
                                             guided=guided,
                                             use_gpu=use_gpu)
        max_gradients = self.calculate_gradients(input_,
                                                 target_class,
                                                 guided=guided,
                                                 take_max=True,
                                                 use_gpu=use_gpu)

        # Setup subplots

        subplots = [
            # (title, [(image1, cmap, alpha), (image2, cmap, alpha)])
            ('Input image',
             [(format_for_plotting(denormalize(input_)), None, None)]),
            ('Gradients across RGB channels',
             [(format_for_plotting(standardize_and_clip(gradients)),
              None,
              None)]),
            ('Max gradients',
             [(format_for_plotting(standardize_and_clip(max_gradients)),
              cmap,
              None)]),
            ('Overlay',
             [(format_for_plotting(denormalize(input_)), None, None),
              (format_for_plotting(standardize_and_clip(max_gradients)),
               cmap,
               alpha)])
        ]

        fig = plt.figure(figsize=figsize)

        for i, (title, images) in enumerate(subplots):
            ax = fig.add_subplot(1, len(subplots), i + 1)
            ax.set_axis_off()
            ax.set_title(title)

            for image, cmap, alpha in images:
                ax.imshow(image, cmap=cmap, alpha=alpha)

        if return_output:
            return gradients, max_gradients

    #####################
    # Private interface #
    #####################

    def _register_conv_hook(self):
        def _record_gradients(module, grad_in, grad_out):
            if self.gradients.shape == grad_in[0].shape:
                self.gradients = grad_in[0]

        for _, module in self.model.named_modules():
            if isinstance(module, nn.modules.conv.Conv2d):
                module.register_backward_hook(_record_gradients)
                break

    def _register_relu_hooks(self):
        def _record_output(module, input_, output):
            self.relu_outputs.append(output)

        def _clip_gradients(module, grad_in, grad_out):
            relu_output = self.relu_outputs.pop()
            clippled_grad_out = grad_out[0].clamp(0.0)

            return (clippled_grad_out.mul(relu_output),)

        for _, module in self.model.named_modules():
            if isinstance(module, nn.ReLU):
                module.register_forward_hook(_record_output)
                module.register_backward_hook(_clip_gradients)
