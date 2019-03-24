"""
"""

import torch

class Backprop:
    """
    """
    def __init__(self, model):
        """
        """
        self.model = model
        self.model.eval()

        self.nodes = list(self.model.children())
        self.target_layer_type = torch.nn.modules.conv.Conv2d
        self.target_layer_in_channels = 3

        self.gradients = None
        self.register_hooks()

    def find_target_layer(self):
        """
        """
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

    def register_hooks(self):
        def record_gradient(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        first_conv_layer = self.find_target_layer()
        first_conv_layer.register_backward_hook(record_gradient)

