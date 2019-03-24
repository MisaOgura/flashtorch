"""
"""
import pdb
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
        self.register_hook()

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

    def register_hook(self):
        """
        """
        def record_gradient(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        first_conv_layer = self.find_target_layer()
        first_conv_layer.register_backward_hook(record_gradient)

    def calculate_gradient(self, inputs, target_class):
        """
        """
        # raw prediction values (logit) from the last linear layer
        output = self.model(inputs)

        self.model.zero_grad()

        # Create a 2D tensor with shape (1, num_classes) and
        # set all element to zero
        target = torch.FloatTensor(1, output.size()[-1]).zero_()

        # Set the element at target class index to be 1
        target[0][target_class] = 1

        # Calculate gradient of the value corresponding to the target class
        # i.e output[target_class] w.r.t inputs

        output.backward(gradient=target)

        return self.gradients
