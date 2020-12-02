# Learnergy: Energy-based Machine Learners

[![Latest release](https://img.shields.io/github/release/gugarosa/learnergy.svg)](https://github.com/gugarosa/learnergy/releases)
[![DOI](https://zenodo.org/badge/189021939.svg)](https://zenodo.org/badge/latestdoi/189021939)
[![Build status](https://img.shields.io/travis/com/gugarosa/learnergy/master.svg)](https://github.com/gugarosa/learnergy/releases)
[![Open issues](https://img.shields.io/github/issues/gugarosa/learnergy.svg)](https://github.com/gugarosa/learnergy/issues)
[![License](https://img.shields.io/github/license/gugarosa/learnergy.svg)](https://github.com/gugarosa/learnergy/blob/master/LICENSE)

## Welcome to Learnergy.

Did you ever reach a bottleneck in your computational experiments? Are you tired of implementing your own techniques? If yes, Learnergy is the real deal! This package provides an easy-to-go implementation of energy-based machine learning algorithms. From datasets to fully-customizable models, from internal functions to external communications, we will foster all research related to energy-based machine learning.

Use Learnergy if you need a library or wish to:

* Create your energy-based machine learning algorithm;
* Design or use pre-loaded learners;
* Mix-and-match different strategies to solve your problem;
* Because it is incredible to learn things.

Read the docs at [learnergy.readthedocs.io](https://learnergy.readthedocs.io).

Learnergy is compatible with: **Python 3.6+**.

---

## Package guidelines

1. The very first information you need is in the very **next** section.
2. **Installing** is also easy if you wish to read the code and bump yourself into, follow along.
3. Note that there might be some **additional** steps in order to use our solutions.
4. If there is a problem, please do not **hesitate**, call us.

---

## Citation

If you use Learnergy to fulfill any of your needs, please cite us:

```BibTex
@misc{roder2020learnergy,
    title={Learnergy: Energy-based Machine Learners},
    author={Mateus Roder and Gustavo Henrique de Rosa and João Paulo Papa},
    year={2020},
    eprint={2003.07443},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

---

## Getting started: 60 seconds with Learnergy

First of all. We have examples. Yes, they are commented. Just browse to `examples/`, choose your subpackage, and follow the example. We have high-level examples for most of the tasks we could think.

Alternatively, if you wish to learn even more, please take a minute:

Learnergy is based on the following structure, and you should pay attention to its tree:

```yaml
- learnergy
    - core
        - dataset
        - model
    - math
        - scale
    - models
        - bernoulli
            - conv_rbm
            - discriminative_rbm
            - dropout_rbm
            - e_dropout_rbm
            - rbm
        - deep
            - conv_dbn
            - dbn
            - residual_dbn
        - extra
            - sigmoid_rbm
        - gaussian
            - gaussian_conv_rbm        
            - gaussian_rbm
    - utils
        - constants
        - exception
        - logging
    - visual
        - image
        - metrics
        - tensor
```

### Core

Core is the core. Essentially, it is the parent of everything. You should find parent classes defining the basis of our structure. They should provide variables and methods that will help to construct other modules.

### Math

Just because we are computing stuff, it does not means that we do not need math. Math is the mathematical package, containing low-level math implementations. From random numbers to distributions generation, you can find your needs on this module.

### Models

This is the heart. All models are declared and implemented here. We will offer you the most fantastic implementation of everything we are working with. Please take a closer look into this package.

### Utils

This is a utility package. Common things shared across the application should be implemented here. It is better to implement once and use as you wish than re-implementing the same thing over and over again.

### Visual

Everyone needs images and plots to help visualize what is happening, correct? This package will provide every visual-related method for you. Check a specific image, your fitness function convergence, plot reconstructions, weights, and much more.

---

## Installation

We believe that everything has to be easy. Not tricky or daunting, Learnergy will be the one-to-go package that you will need, from the very first installation to the daily-tasks implementing needs. If you may just run the following under your most preferred Python environment (raw, conda, virtualenv, whatever):

```bash
pip install learnergy
```

Alternatively, if you prefer to install the bleeding-edge version, please clone this repository and use:

```bash
pip install -e .
```

---

## Environment configuration

Note that sometimes, there is a need for additional implementation. If needed, from here, you will be the one to know all of its details.

### Ubuntu

No specific additional commands needed.

### Windows

No specific additional commands needed.

### MacOS

No specific additional commands needed.

---

## Support

We know that we do our best, but it is inevitable to acknowledge that we make mistakes. If you ever need to report a bug, report a problem, talk to us, please do so! We will be available at our bests at this repository or mateus.roder@unesp.br and gustavo.rosa@unesp.br.

---
