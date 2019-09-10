# FlashTorch

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/flashtorch.svg?color=green)](https://pypi.org/project/flashtorch/)
[![PyPI](https://img.shields.io/pypi/v/flashtorch.svg?color=yellow)](https://pypi.org/project/flashtorch/)
[![PyPI - License](https://img.shields.io/pypi/l/flashtorch.svg?color=black)](https://github.com/MisaOgura/flashtorch/blob/master/LICENSE)
[![DOI](https://zenodo.org/badge/177140934.svg)](https://zenodo.org/badge/latestdoi/177140934)
[![Say Thanks!](https://img.shields.io/badge/Say%20Thanks-!-1EAEDB.svg)](https://saythanks.io/to/MisaOgura)

A Python visualizaion toolkit for neural networks in PyTorch.

You can apply feature visualization techniques (such as **saliency maps** and **activation maximization**) on your model, with as little as a few lines of code!

It is compatible with pre-trained models that come with [torchvision](https://pytorch.org/docs/stable/torchvision/models.html), and seemlessly integrates with other custom models built in PyTorch.

## Overview

- [Installation](#installation)
- [API guide](#api-guide)
- [How to use](#how-to-use)
  - [Image handling](#image-handling-flashtorchutils)
  - [Saliency maps](#saliency-maps-flashtorchsaliency)
  - [Activation maximization](#activation-maximization-flashtorchactivmax)
- [Talks & blog posts](#talks--blog-posts)
- [Reading](#reading)
- [Inspiration](#inspiration)
- [Citation](#citation)
- [Author](#author)

## Installation

If you are installing `flashtorch` for the first time:

```bash
$ pip install flashtorch
```

Or if you are upgrading it:

```bash
$ pip install flashtorch -U
```

## API guide

These are currently available modules.

- `flashtorch.utils`: some useful utility functions for data handling & transformation
- `flashtorch.utils.imagenet`: `ImageNetIndex` class for easy-ish retrieval of class index
- `flashtorch.saliency.backprop`: `Backprop` class for calculating gradients
- `flashtorch.activmax.gradient_ascent`: `GradientAscent` class for activation maximization

You can inspect each module with Python built-in function `help`. The output of that is available on [Quick API Guide](https://github.com/MisaOgura/flashtorch/wiki/Quick-API-Guide) for your convenience.

## How to use

Below, you can find simple demos to get you started, as well as links to some handy notebooks showing additional examples of using `flashtorch`.

### Image handling (`flashtorch.utils`)

- [Image handling](https://github.com/MisaOgura/flashtorch/blob/master/examples/examples/image_handling.ipynb) notebook

### Saliency maps (`flashtorch.saliency`)

- [Saliency map with backpropagation](https://github.com/MisaOgura/flashtorch/blob/master/examples/visualize_saliency_with_backprop.ipynb) notebook
- [Google Colab](https://colab.research.google.com/github/MisaOgura/flashtorch/blob/master/examples/visualize_saliency_with_backprop_colab.ipynb) version - best for trying it out

**[Saliency](https://en.wikipedia.org/wiki/Salience_(neuroscience))** in human visual perception is a _subjective quality_ that makes certain things within the field of view _stand out_ from the rest and _grabs our attention_.

**Saliency maps** in computer vision provide indications of the most salient regions within images. By creating a saliency map for neural networks, we can gain some intuition on _"where the network is paying the most attention to"_ in an imput image.

Using `flashtorch.saliency` module, let's visualise image-specific class saliency maps of [AlexNet](https://arxiv.org/abs/1404.5997) pre-trained on [ImageNet](http://www.image-net.org/) classification tasks.

![Saliency map of great grey owl in Alexnet](https://github.com/MisaOgura/flashtorch/blob/master/examples/images/saliency_demo.png)

The network is focusing on the sunken eyes and the round head for this owl.

### Activation maximization (`flashtorch.activmax`)

- [Activation maximization](https://github.com/MisaOgura/flashtorch/blob/master/examples/activation_maximization.ipynb) notebook
- [Google Colab](https://colab.research.google.com/github/MisaOgura/flashtorch/blob/master/examples/activation_maximization_colab.ipynb) version - best for trying it out

[Activation maximization](https://pdfs.semanticscholar.org/65d9/94fb778a8d9e0f632659fb33a082949a50d3.pdf) is one form of feature visualization that allows us to visualize what CNN filters are "looking for", by applying each filter to an input image and updating the input image so as to maximize the activation of the filter of interest (i.e. treating it as a gradient ascent task with filter activation values as the loss).

Using `flashtorch.activmax` module, let's visualize images optimized with filters
from [VGG16](https://arxiv.org/pdf/1409.1556.pdf) pre-trained on [ImageNet](http://www.image-net.org/) classification tasks.

![VGG16 conv5_1 filters](https://github.com/MisaOgura/flashtorch/blob/master/examples/images/activmax_demo.png)

Concepts such as _'eyes'_ (filter 45) and _'entrances (?)'_ (filter 271) seem to appear in the conv5_1 layer of VGG16.

Visit the notebook above to see what earlier layers do!

## Talks & blog posts

- [Hopperx1 London](http://www.cvent.com/events/hopperx1-london/agenda-e7d0f2fa5e9d46cf88fd8c322ae1290b.aspx), June 2019 - [slide deck](https://misaogura.github.io/flashtorch/presentations/Hopperx1London)

- [Uncovering what neural nets “see” with FlashTorch](https://towardsdatascience.com/feature-visualisation-in-pytorch-saliency-maps-a3f99d08f78a)

- [Gaining insights on transfer learning with FlashTorch](https://towardsdatascience.com/gaining-insights-on-transfer-learning-with-flashtorch-de344df0f410)

## Reading

- Introduction and overview of feature visualizaion: [Feature Visualization](https://distill.pub/2017/feature-visualization/)

- The latest development in feature visualizaion: [Exploring Neural Networks with Activation Atlases](https://distill.pub/2019/activation-atlas/)

- Using backpropagation for gradient visualizaion: [Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps](https://arxiv.org/pdf/1312.6034.pdf)

- Guided backprobagation: [Striving for Simplicity: The All Convolutional Net](https://arxiv.org/pdf/1412.6806.pdf)

## Inspiration

- [pytorch-cnn-visualizations](https://github.com/utkuozbulak/pytorch-cnn-visualizations) by utkuozbulak

- [keras-vis](https://github.com/raghakot/keras-vis) by raghakot

## Citation

```txt
Misa Ogura. (2019, September 9).
MisaOgura/flashtorch: 0.1.0 (Version v0.1.0).
Zenodo. http://doi.org/10.5281/zenodo.3403214
```

## Author

### Misa Ogura

#### R&D Software Engineer @ [BBC](https://www.bbc.co.uk/rd/blog)

#### Co-founder of [Women Driven Development](https://womendrivendev.org/)

[Github](https://github.com/MisaOgura) | [Medium](https://medium.com/@misaogura) | [twitter](https://twitter.com/misa_ogura) | [LinkedIn](https://www.linkedin.com/in/misaogura/)
