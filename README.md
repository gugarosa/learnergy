# Learnergy: Energy-based Machine Learners

[![Latest release](https://img.shields.io/github/release/gugarosa/learnergy.svg)](https://github.com/gugarosa/learnergy/releases)
[![CI](https://github.com/gugarosa/learnergy/actions/workflows/ci.yml/badge.svg)](https://github.com/gugarosa/learnergy/actions/workflows/ci.yml)
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.4390744-006DB9.svg)](https://doi.org/10.5281/zenodo.4390744)
[![License](https://img.shields.io/github/license/gugarosa/learnergy.svg)](LICENSE)

Learnergy provides PyTorch implementations of Restricted Boltzmann Machines
(RBMs) and Deep Belief Networks (DBNs) for unsupervised feature learning,
generative modeling, and classification. Recurrent temporal variants model
sequences with hidden-state recurrence. The package also includes dataset
adapters, image-quality metrics, and visualization helpers.

## Installation

Learnergy requires Python 3.11 or newer. Add it to a project managed by uv with:

```bash
uv add learnergy
```

Add the optional torchvision dependency to run the examples:

```bash
uv add "learnergy[examples]"
```

For a consumer installation in an existing Python environment, pip is also supported:

```bash
pip install learnergy
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
| Temporal | `RTRBM`, `RTGaussianRBM`, `RTVarianceGaussianRBM`, `RTDBN` |

The `learnergy.core.Dataset`, `learnergy.math`, and `learnergy.visual` modules
remain available for array-backed datasets, SSIM/scaling helpers, convergence
plots, image mosaics, and tensor rendering.

See [`examples/applications`](examples/applications) for complete training and
classification programs.

### Temporal models

Temporal RBMs consume floating-point tensors shaped
`(batch_size, sequence_length, n_visible)`. Dataset items pair a sequence with
an ignored target. Each sequence starts from a learned initial context, and
hidden probabilities carry the recurrence between timesteps:

```python
import torch
from torch.utils.data import TensorDataset

from learnergy.models.temporal import RTRBM

sequences = torch.bernoulli(torch.rand(32, 6, 4))
dataset = TensorDataset(sequences, torch.zeros(32))

model = RTRBM(n_visible=4, n_hidden=8, learning_rate=0.01)

mse = model.fit(dataset, batch_size=8, epochs=2)
hidden_sequences = model(sequences)
generated = model.sample(n_samples=3, n_steps=6, gibbs_steps=20)
```

`RTDBN` supports Bernoulli, fixed-variance Gaussian, and learned-variance
Gaussian layers, returning mean-pooled sequence embeddings. See the
[temporal-model guide](docs/temporal.rst) for normalization, sampling, warmup,
gradient, and checkpoint contracts, or run the
[self-contained temporal example](examples/applications/temporal/rtrbm_training.py).

### Numerical behavior

When enabled, Gaussian normalization uses statistics from the current batch,
not stored training statistics. Batches of two or more samples use sample
standard deviation; a singleton batch is centered to zero. Representations
therefore depend on batch composition. Disable the corresponding normalization
flags when supplying externally standardized features.

`VarianceGaussianRBM.sigma` is a learnable scale: the effective visible variance
is `sigma**2` plus a dtype-dependent epsilon. Its `visible_sampling` method
returns conditional means followed by sampled states, and Gibbs sampling uses
those states.

Gaussian convolutional representations support gradient-based fine-tuning.
Use `torch.no_grad()` when extracting frozen features without an autograd graph.

The corrected variance-Gaussian sampling and stabilized likelihood calculations
can change training trajectories, including with a fixed random seed.

`RTVarianceGaussianRBM` follows the same variance and sampler tuple-order
conventions as `VarianceGaussianRBM`. Fixed-variance `RTGaussianRBM` follows
`GaussianRBM` for deterministic training conditionals and adds unit Gaussian
noise during generation. Its optional normalization pools batch and time
dimensions, handles singleton observations, and detaches standardized inputs.

## Development

Follow [the coding conventions](CONVENTIONS.md) when changing the library or its examples.

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
