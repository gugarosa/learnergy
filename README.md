# Learnergy: Energy-based Machine Learners

[![Latest release](https://img.shields.io/github/release/gugarosa/learnergy.svg)](https://github.com/gugarosa/learnergy/releases)
[![DOI](http://img.shields.io/badge/DOI-10.5281/zenodo.4390744-006DB9.svg)](https://doi.org/10.5281/zenodo.4390744)
[![Open issues](https://img.shields.io/github/issues/gugarosa/learnergy.svg)](https://github.com/gugarosa/learnergy/issues)
[![License](https://img.shields.io/github/license/gugarosa/learnergy.svg)](https://github.com/gugarosa/learnergy/blob/master/LICENSE)

## Welcome to Learnergy

Learnergy is a PyTorch-based framework for **energy-based machine learning**, providing ready-to-use implementations of Restricted Boltzmann Machines (RBMs) and Deep Belief Networks (DBNs). It is designed for researchers and practitioners who need a clean, modular library for unsupervised feature learning, generative modeling, and classification with energy-based models.

### What you can do

- **Train RBMs** with various unit types: Bernoulli, Gaussian, Sigmoid, ReLU, SeLU
- **Apply regularization**: Dropout, DropConnect, and Energy-based Dropout
- **Build deep architectures**: stack RBMs into DBNs and Convolutional DBNs
- **Use residual learning**: ResidualDBN with skip connections for improved information flow
- **Classify**: Discriminative and Hybrid Discriminative RBMs for supervised tasks
- **Visualize**: convergence plots, weight mosaics, and tensor images

### Quick start

```python
import torchvision
from learnergy.models.bernoulli import RBM

# Load MNIST
train = torchvision.datasets.MNIST(
    root="./data", train=True, download=True,
    transform=torchvision.transforms.ToTensor(),
)

# Train a Bernoulli RBM
model = RBM(n_visible=784, n_hidden=128, steps=1, learning_rate=0.1)
mse, pl = model.fit(train, batch_size=128, epochs=5)

# Reconstruct
rec_mse, visible_probs = model.reconstruct(train)
```

For a Gaussian RBM with continuous inputs:

```python
from learnergy.models.gaussian import GaussianRBM

model = GaussianRBM(n_visible=784, n_hidden=256, steps=1, learning_rate=0.005)
mse, pl = model.fit(train, batch_size=128, epochs=10)
```

For a Deep Belief Network:

```python
from learnergy.models.deep import DBN

model = DBN(
    model=("gaussian", "sigmoid"),
    n_visible=784, n_hidden=(256, 128),
    steps=(1, 1), learning_rate=(0.01, 0.01),
    momentum=(0, 0), decay=(0, 0), temperature=(1, 1),
)
mse, pl = model.fit(train, batch_size=128, epochs=(5, 5))
```

Browse the `examples/` directory for more use cases, including classification, convolutional models, and fine-tuning.

Learnergy is compatible with: **Python 3.9+** and **PyTorch 1.8+**.

---

## Architecture

For a detailed walkthrough of the codebase design, class hierarchy, and design patterns, see [ARCHITECTURE.md](ARCHITECTURE.md).

```
learnergy/
├── core/          # Dataset and Model base classes
├── math/          # SSIM metrics, scaling utilities
├── models/
│   ├── bernoulli/ # RBM, ConvRBM, DiscriminativeRBM, Dropout/DropConnect, EDropout
│   ├── gaussian/  # GaussianRBM (+ ReLU, SeLU, Variance), GaussianConvRBM
│   ├── extra/     # SigmoidRBM
│   └── deep/      # DBN, ConvDBN, ResidualDBN
├── utils/         # Constants, custom exceptions, logging
└── visual/        # Convergence plots, image mosaics, tensor display
```

### Available models

| Family | Models |
|---|---|
| **Bernoulli** | `RBM`, `ConvRBM`, `DiscriminativeRBM`, `HybridDiscriminativeRBM`, `DropoutRBM`, `DropConnectRBM`, `EDropoutRBM` |
| **Gaussian** | `GaussianRBM`, `GaussianReluRBM`, `GaussianSeluRBM`, `VarianceGaussianRBM`, `GaussianConvRBM` |
| **Extra** | `SigmoidRBM` |
| **Deep** | `DBN`, `ConvDBN`, `ResidualDBN` |

---

## Installation

```bash
pip install learnergy
```

Or install from source for the latest version:

```bash
git clone https://github.com/gugarosa/learnergy.git
cd learnergy
pip install -e .
```

### Dependencies

| Package | Version | Purpose |
|---|---|---|
| PyTorch | ≥ 1.8.0 | Core tensor operations and GPU support |
| torchvision | ≥ 0.9.0 | Dataset loading and transforms |
| matplotlib | ≥ 3.3.4 | Visualization |
| Pillow | ≥ 8.1.2 | Image mosaic creation |
| scikit-image | ≥ 0.17.2 | SSIM metric |
| tqdm | ≥ 4.49.0 | Progress bars |

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

## Support

If you need to report a bug or have questions, please open an [issue](https://github.com/gugarosa/learnergy/issues) or reach out at mateus.roder@unesp.br and gustavo.rosa@unesp.br.

---
