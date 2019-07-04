# Recogners: Python-Inspired Machine Learners

[![Latest release](https://img.shields.io/github/release/recogna-lab/recogners.svg)](https://github.com/recogna-lab/recogners/releases)
[![Build status](https://img.shields.io/travis/com/recogna-lab/recogners/master.svg)](https://github.com/recogna-lab/recogners/releases)
[![Open issues](https://img.shields.io/github/issues/recogna-lab/recogners.svg)](https://github.com/recogna-lab/recogners/issues)
[![License](https://img.shields.io/github/license/recogna-lab/recogners.svg)](https://github.com/recogna-lab/recogners/blob/master/LICENSE)

## Welcome to Recogners.
Did you ever reach a bottleneck in your computational experiments? Are you tired of implementing your own techniques? If yes, Recogners is the real deal! This package provides an easy-to-go implementation of machine learning algorithms. From datasets to fully-customizable models, from internal functions to external communication, we will foster all research related to machine learning.

Use Recogners if you need a library or wish to:
* Create your own machine learning algorithm.
* Design or use pre-loaded learners.
* Mix-and-match different strategies to solve your problem.
* Because it is incredible to learn things.

Read the docs at [recogners.readthedocs.io](https://recogners.readthedocs.io).

Recogners is compatible with: **Python 3.6 and 3.7**.

---

## Package guidelines

1. The very first information you need is in the very **next** section.
2. **Installing** is also easy, if you wish to read the code and bump yourself into, just follow along.
3. Note that there might be some **additional** steps in order to use our solutions.
4. If there is a problem, please do not **hesitate**, call us.

---

## Getting started: 60 seconds with recogners

First of all. We have examples. Yes, they are commented. Just browse to `examples/`, chose your subpackage and follow the example. We have high-level examples for most tasks we could think of.

Or if you wish to learn even more, please take a minute:

Recogners is based on the following structure, and you should pay attention to its tree:

```
- recogners
    - core
        - dataset
        - model
    - datasets
        - opf
    - math
        - scale
    - models
        - rbm
    - utils
        - loader
        - logging
    - visual
        - image
        - plot
```

### Core

Core is the core. Essentially, it is the parent of everything. You should find parent classes defining the basic of our structure. They should provide variables and methods that will help to construct other modules.

### Datasets

As we are build over PyTorch, we will also provide some integrations with another techniques used by our laboratory. Initially, you are able to load all OPF text files into this library.

### Math

Just because we are computing stuff, it does not means that we do not need math. Math is the mathematical package, containing low level math implementations. From random numbers to distributions generation, you can find your needs on this module.

### Models

This is the heart, basically. All models are declared and implemented here. We will offer you the most amazing implementation of everything we are working with. Please take a closer look into this package.

### Utils

This is an utilities package. Common things shared across the application should be implemented here. It is better to implement once and use as you wish than re-implementing the same thing over and over again.

### Visual

Every one needs images and plots to help visualize what is happening, correct? This package will provide every visual-related method for you. Check a specific image, your fitness function convergence, plot reconstructions, weights and much more!

---

## Installation

We belive that everything have to be easy. Not difficult or daunting, Recogners will be the one-to-go package that you will need, from the very first instalattion to the daily-tasks implementing needs. If you may, just run the following under your most preferred Python environment (raw, conda, virtualenv, whatever):

```Python
pip install recogners
```

Or, if you prefer to install the bleeding-edge version, please clone this repository and use:

```Python
pip install .
```

---

## Environment configuration

Note that sometimes, there is a need for an additional implementation. If needed, from here you will be the one to know all of its details.

### Ubuntu

No specific additional commands needed.

### Windows

No specific additional commands needed.

### MacOS

No specific additional commands needed.

---

## Support

We know that we do our best, but it's inevitable to acknowlodge that we make mistakes. If you every need to report a bug, report a problem, talk to us, please do so! We will be avaliable at our bests at this repository or recogna@fc.unesp.br.

---
