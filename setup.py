#!/usr/bin/env python
""" FlashTorch is a feature visualization toolkit.
It is built with PyTorch, for neural networks in PyTorch.

Neural networks are often described as "black box". The lack of understanding
on how neural networks make predictions enables unpredictable/biased models,
causing real harm to society and a loss of trust in AI-assisted systems.

Feature visualization is an area of research, which aims to understand how
neural networks perceive images. However, implementing such techniques is often
complicated.

FlashTorch was created to solve this problem!

You can apply feature visualization techniques such as saliency maps and
activation maximization on your model, with as little as a few lines of code.

It is compatible with pre-trained models that come with torchvision, and
seamlessly integrates with other custom models built in PyTorch.

All FlashTorch wheels on PyPI are distrubuted with the MIT License.
"""

from setuptools import setup, find_packages

DOCLINES = (__doc__ or '').split("\n")
long_description = "\n".join(DOCLINES[2:])

version = '0.1.2'

setup(
    name='flashtorch',
    version=version,
    author='Misa Ogura',
    author_email='misa.ogura01@gmail.com',
    description='Visualization toolkit for neural networks in PyTorch',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/MisaOgura/flashtorch',
    packages=find_packages(exclude=['tests*']),
    include_package_data=True,
    package_data={'flashtorch.utils.resources':
        ['imagenet_class_index.json']
    },
    install_requires=[
        'matplotlib',
        'numpy',
        'Pillow',
        'torch',
        'torchvision',
        'importlib_resources'
    ],
    classifiers=[
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
