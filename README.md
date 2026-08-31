# Learnergy: Energy-based Machine Learners

[![Latest release](https://img.shields.io/github/release/gugarosa/learnergy.svg)](https://github.com/gugarosa/learnergy/releases)
[![CI](https://github.com/gugarosa/learnergy/actions/workflows/ci.yml/badge.svg)](https://github.com/gugarosa/learnergy/actions/workflows/ci.yml)
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.4390744-006DB9.svg)](https://doi.org/10.5281/zenodo.4390744)
[![License](https://img.shields.io/github/license/gugarosa/learnergy.svg)](LICENSE)

Learnergy provides PyTorch implementations of Restricted Boltzmann Machines
(RBMs) and Deep Belief Networks (DBNs) for unsupervised feature learning,
generative modeling, and classification. It also includes dataset adapters,
image-quality metrics, and visualization helpers.

## Installation

Learnergy requires Python 3.11 or newer.

```bash
pip install learnergy
```

Install the optional torchvision dependency to run the examples:

```bash
pip install "learnergy[examples]"
```

## Quick start

```python
import torch
from torch.utils.data import TensorDataset

from learnergy.models.bernoulli import RBM

samples = torch.bernoulli(torch.rand(1_024, 784))
targets = torch.zeros(1_024)
dataset = TensorDataset(samples, targets)

model = RBM(n_visible=784, n_hidden=128, learning_rate=0.1)
mse, pseudo_likelihood = model.fit(dataset, batch_size=128, epochs=5)
reconstruction_mse, reconstructed = model.reconstruct(dataset)
```

Stack RBMs into a DBN:

```python
from learnergy.models.deep import DBN

model = DBN(
    model=("gaussian", "sigmoid"),
    n_visible=784,
    n_hidden=(256, 128),
    steps=(1, 1),
    learning_rate=(0.01, 0.01),
    momentum=(0, 0),
    decay=(0, 0),
    temperature=(1, 1),
)
model.fit(dataset, batch_size=128, epochs=(5, 5))
```

## Available models

| Family | Models |
|---|---|
| Bernoulli | `RBM`, `ConvRBM`, `DiscriminativeRBM`, `HybridDiscriminativeRBM`, `DropoutRBM`, `DropConnectRBM`, `EDropoutRBM` |
| Gaussian | `GaussianRBM`, `GaussianReluRBM`, `GaussianSeluRBM`, `VarianceGaussianRBM`, `GaussianConvRBM` |
| Extra | `SigmoidRBM` |
| Deep | `DBN`, `ConvDBN`, `ResidualDBN` |

The `learnergy.core.Dataset`, `learnergy.math`, and `learnergy.visual` modules
remain available for array-backed datasets, SSIM/scaling helpers, convergence
plots, image mosaics, and tensor rendering.

See [`examples/applications`](examples/applications) for complete training and
classification programs.

## Development

The repository uses [uv](https://docs.astral.sh/uv/) for reproducible
environments and packaging:

```bash
uv sync --locked
uv run pytest
uv build
```

## Citation

```bibtex
@misc{roder2020learnergy,
    title={Learnergy: Energy-based Machine Learners},
    author={Mateus Roder and Gustavo Henrique de Rosa and João Paulo Papa},
    year={2020},
    eprint={2003.07443},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

## Support

Open an [issue](https://github.com/gugarosa/learnergy/issues) for bug reports
and questions.
