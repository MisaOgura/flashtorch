import argparse
from pathlib import Path
import torch
from torchvision import models, transforms
import matplotlib.pyplot as plt
from PIL import Image

from flashtorch.utils import apply_transforms, load_image
from flashtorch.saliency import Backprop
from flashtorch.utils import (denormalize,
                              format_for_plotting,
                              standardize_and_clip)

parser = argparse.ArgumentParser()
parser.add_argument('--img', required=True,
                    help='Path to the image to be found activations on')
parser.add_argument('--target', required=True,
                    help='The target class to find activations on')
parser.add_argument('--model', required=True,
                    default='resnet18',
                    help='type of architecture. string must match pytorch zoo')


opt = parser.parse_args()
print(opt)


class myBackprop(Backprop):
    '''
    This methid is essentially the same as flastorch Backprop
    with the modification that it does not normalize the input image
    while displaying the plot
    '''

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
        """

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
             [(format_for_plotting(input_), None, None)]),
            ('Gradients across RGB channels',
             [(format_for_plotting(standardize_and_clip(gradients)),
              None,
              None)]),
            ('Max gradients',
             [(format_for_plotting(standardize_and_clip(max_gradients)),
              cmap,
              None)]),
            ('Overlay',
             [(format_for_plotting(input_), None, None),
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


def load_model(model_name='resnet18', ptrained=True):
    known_models = [x for x in dir(models)]
    if model_name not in known_models:
        raise ValueError('specified model doesnt exist in pytorch zoo')

    # This is equivalent to calling models.model_name(pretrained=True)
    # e.g models.alexnet(pretrained=True)
    model = getattr(models, model_name)(pretrained=ptrained)

    model.eval()
    return model


if __name__ == '__main__':
    if torch.cuda.is_available():
        use_gpu = True
        device = "cuda:0"
    else:
        use_gpu = False
        device = "cpu"

    # Load a model from pytorch-zoo
    # choose from any of
    '''
    names  = ['alexnet', 'vgg16',
              'resnet18', 'resnet34', 'resnet50',
              'squeezenet1_0', 'densenet161', 'inception_v3',
              'googlenet', 'shufflenet_v2_x1_0', 'mobilenet_v2']
              and more in https://pytorch.org/docs/stable/torchvision/models.html
    '''
    model = load_model(opt.model)
    model.to(device)

    backprop = myBackprop(model)
    backprop.use_gpu = use_gpu

    infer_img = Image.open(opt.img).convert('RGB')

    ''' If your input image is different size
    from the model head uncomment and use these transforms'''
    # means = [0.485, 0.456, 0.406]
    # stds =  [0.229, 0.224, 0.225]

    transform = transforms.Compose([
                                    # transforms.Resize(size),
                                    # transforms.CenterCrop(size),
                                    transforms.ToTensor()
                                    # transforms.Normalize(means, stds)
                                    ])

    tensor = transform(infer_img).unsqueeze(0)
    tensor.requires_grad = True
    tensor.to(device)

    output = backprop.visualize(tensor, int(opt.target),
                                guided=True, use_gpu=backprop.use_gpu)
    f_n = Path(opt.img).stem
    plt.suptitle('{} - Guided Backpropogation'.format(opt.model))
    plt.savefig('result-{}-backprop-{}.png'.format(f_n, opt.model))
