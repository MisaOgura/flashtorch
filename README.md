# FlashTorch

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/flashtorch.svg?color=green)](https://pypi.org/project/flashtorch/)
[![PyPI](https://img.shields.io/pypi/v/flashtorch.svg?color=yellow)](https://pypi.org/project/flashtorch/)
[![PyPI - License](https://img.shields.io/pypi/l/flashtorch.svg?color=black)](https://github.com/MisaOgura/flashtorch/blob/master/LICENSE)
[![DOI](https://zenodo.org/badge/177140934.svg)](https://zenodo.org/badge/latestdoi/177140934)

Visualisation toolkit implemented in PyTorch for inspecting what neural networks learn in image recognition tasks (feature visualisation).

The project is very much work in progress, and I would appreciate your feedback!

It currently supports visualisation of saliancy maps for all the models available under [torchvision.models](https://pytorch.org/docs/stable/torchvision/models.html).

## Overview

- [Installation](#installation)
- [API guide](#api-guide)
- [Example notebooks](#example-notebooks)
  - [Image handling](#image-handling)
  - [Saliency maps](#saliency-maps)
- [Talks & blog posts](#talks--blog-posts)
- [Papers](#papers)
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

An API guide is under construction, so this is a temporary workaround.

These are currently available modules.

- `flashtorch.utils`: some useful utilility functions for data handling & transformation
- `flashtorch.utils.imagenet`: `ImageNetIndex` class for easy-ish retrieval of class index
- `flashtorch.saliency.backprop`: `Backprop` class for calculating gradients

You can inspect each module with Python built-in function `help`. The output of that is available on [Quick API Guide page](https://github.com/MisaOgura/flashtorch/wiki/Quick-API-Guide) for your convenience.

## Example notebooks

Here are some handy notebooks showing you examples of using `flashtorch`.

### Image handling

Notebook: [Image handling](./examples/image_handling.ipynb)

### Saliency maps

Notebook: [Image-specific class saliency map with backpropagation](./examples/visualise_saliency_with_backprop.ipynb)

  - Notebook also available on [Google Colab](https://colab.research.google.com/github/MisaOgura/flashtorch/blob/master/examples/visualise_saliency_with_backprop_colab.ipynb) - probably the best way to play around quickly, as there is no need for setting up the environment!

**[Saliency](https://en.wikipedia.org/wiki/Salience_(neuroscience))** in human visual perception is a _subjective quality_ that makes certain things within the field of view _stand out_ from the rest and _grabs our attention_.

**Saliency maps** in computer vision provide indications of the most salient regions within images. By creating a saliency map for neural networks, we can gain some intuition on _"where the network is paying the most attention to"_ in an imput image.

#### AlexNet visualisation

Using `flashtorch.saliency` module, let's visualise image-specific class saliency maps of [AlexNet](https://arxiv.org/abs/1404.5997) pre-trained on [ImageNet](http://www.image-net.org/) classification tasks.

**Great gray owl** (class index 24):
The network is focusing on the sunken eyes and the round head for this owl.

![Saliency map of great grey owl in Alexnet](examples/images/alexnet_great_grey_owl.png)

**Peacock** (class index 84):
But it doesn't always focus on the eyes and head of an animal. In its world's view, what makes peacok a peacok is the eyespots on its tail!

![Saliency map of peacock in Alexnet](examples/images/alexnet_peacock.png)

**Toucan** (class index 96):
And in case of a toucan, the network is paying an intense attention on its beak.

![Saliency map of tucan in Alexnet](examples/images/alexnet_tucan.png)

Do you agree?

#### Insignts on transfer learning

In the example above, we've visualised saliency maps for a network that has been trained on ImageNet and used images of objects which it _already knows_.

We can take a step further and investigate _how the network's perception changes_ before and after the training, when presented by a new object.

This time, I'm going to use [DenseNet](https://arxiv.org/abs/1608.06993), which is again pre-trained on ImageNet (1000 classes), and train it into a flower classifier to recognise 102 species of flowers ([dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)).

With _no additional training_, and just by swapping out the last fully-connected layer, the model performs very poorly (0.1% test accuracy). By plotting the gradients, we can see that the network is mainly focusing on the shape of the flower.

**Foxgloves** as an example:

![Transfer learning pre](examples/images/transfer_learning_pre.png)

With training, the model now achieves 98.7% test accuracy. But _why_? What is it that it's seeing now, that it wasn't before?

The network has _learnt to shift its focus_ on the mottle patten within flower cups! In it's world's view, that is the most distinguishing things about this object, which I think closely align with what _we_ deem the most unique trait of this flower.

![Transfer learning post](examples/images/transfer_learning_post.png)

## Talks & blog posts

- [Hopperx1 London](http://www.cvent.com/events/hopperx1-london/agenda-e7d0f2fa5e9d46cf88fd8c322ae1290b.aspx), June 2019 - [slide deck](https://misaogura.github.io/flashtorch/presentations/Hopperx1London)

- [Uncovering what neural nets “see” with FlashTorch](https://towardsdatascience.com/feature-visualisation-in-pytorch-saliency-maps-a3f99d08f78a)

- [Gaining insights on transfer learning with FlashTorch](https://towardsdatascience.com/gaining-insights-on-transfer-learning-with-flashtorch-de344df0f410)

## Papers

- Introduction and overview of feature visualisation: [Feature Visualization](https://distill.pub/2017/feature-visualization/)

- Latest development in feature visualisation: [Exploring Neural Networks with Activation Atlases](https://distill.pub/2019/activation-atlas/)

- Using backpropagation for gradient visualisation: [Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps](https://arxiv.org/pdf/1312.6034.pdf)

- Guided backprobagation: [Striving for Simplicity: The All Convolutional Net](https://arxiv.org/pdf/1412.6806.pdf)

## Inspiration

- [pytorch-cnn-visualizations](https://github.com/utkuozbulak/pytorch-cnn-visualizations) by utkuozbulak

- [keras-vis](https://github.com/raghakot/keras-vis) by raghakot

## Citation

```txt
Misa Ogura. (2019, July 8). MisaOgura/flashtorch: 0.0.8 (Version v0.0.8). Zenodo. http://doi.org/10.5281/zenodo.3271410
```

[![Say Thanks!](https://img.shields.io/badge/Say%20Thanks-!-1EAEDB.svg)](https://saythanks.io/to/MisaOgura)

## Author

### Misa Ogura

#### R&D Software Engineer @ [BBC](https://www.bbc.co.uk/rd/blog)

#### Co-founder of [Women Driven Development](https://womendrivendev.org/)

[Github](https://github.com/MisaOgura) | [Medium](https://medium.com/@misaogura) | [twitter](https://twitter.com/misa_ogura) | [LinkedIn](https://www.linkedin.com/in/misaogura/)
