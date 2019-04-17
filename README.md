# TorchScope :microscope:

Visualisation toolkit for inspecting neural networks with PyTorch.

The project is work in progress, and I would appreciate your feedback.

It currently supports visualisation of saliancy maps for all the models available under [torchvision.models](https://pytorch.org/docs/stable/torchvision/models.html).

## Installation

```bash
$ (sudo) pip install torchscope
```

## Usage (example notebooks)

- [Image handling](./examples/image_handling.ipynb)

- [Image-specific class saliency map with backpropagation](./examples/visualise_saliency_with_backprop.ipynb)

## Papers

- Using backpropagation for gradient visualisation: [Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps](https://arxiv.org/pdf/1312.6034.pdf)

- Guided backprobagation: [Striving for Simplicity: The All Convolutional Net](https://arxiv.org/pdf/1412.6806.pdf)

## Inspirations

- [pytorch-cnn-visualizations](https://github.com/utkuozbulak/pytorch-cnn-visualizations) by utkuozbulak

- [keras-vis](https://github.com/raghakot/keras-vis) by raghakot
