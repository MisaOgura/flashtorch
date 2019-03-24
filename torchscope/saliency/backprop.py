"""
"""
import pdb
import torch

class Backprop:
    """Provides an interface to perform backpropagation.

    This class provids a way to calculate a gradient of a target class output
    w.r.t. an input image, by performing a single backprobagation.

    The gradient obtained can be used to visualise a class-specific saliency
    map, which can gives some intuition on regions within the input image that
    contribute the most (and least) to the corresponding output.

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

        self.nodes = list(self.model.children())

        self.target_layer_type = torch.nn.modules.conv.Conv2d
        self.target_layer_in_channels = 3
        self.gradient = None

        self._register_hook()
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def calculate_gradient(self, input_, target_class):
        """Calculates gradient of the target_class output w.r.t. an input_.

        Args:
            input_ (torch.Tensor): With shape :math:`(N, C, H, W)`.
            target_class (int)

        Returns:
            gradient (torch.Tensor): With shape :math:`(N, C, H, W)`.

        """
        self.model = self.model.to(self._device)
        self.model.zero_grad()

        input_ = input_.to(self._device)
        input_.requires_grad = True

        self.gradient = torch.zeros(input_.shape)

        # Get a raw prediction value (logit) from the last linear layer

        output = self.model(input_)

        # Create a 2D tensor with shape (1, num_classes) and
        # set all element to zero

        target = torch.FloatTensor(1, output.shape[-1]).zero_()

        # Set the element at target class index to be 1

        target[0][target_class] = 1

        # Calculate gradient of the target class output w.r.t. input_

        output.backward(gradient=target)

        return self.gradient.detach().cpu()

    def _find_target_layer(self):
        def search(nodes):
            for node in nodes:
                children = list(node.children())
                is_parent = len(children) > 0

                if is_parent:
                    return search(children)
                else:
                    if type(node) == self.target_layer_type and \
                            node.in_channels == self.target_layer_in_channels:
                        return node

        return search(self.nodes)

    def _register_hook(self):
        def _record_gradient(module, grad_in, grad_out):
            if self.gradient.shape == grad_in[0].shape:
                self.gradient = grad_in[0]

        first_conv_layer = self._find_target_layer()
        first_conv_layer.register_backward_hook(_record_gradient)
