# Learnergy — Architecture Guide

> **Version:** 1.1.5 · **License:** Apache 2.0 · **Python:** 3.9+
> Energy-based Machine Learners built on PyTorch.

---

## 1. Overview

Learnergy is a modular Python framework for energy-based machine learning, centered around **Restricted Boltzmann Machines (RBMs)** and their deep extensions — **Deep Belief Networks (DBNs)**. Built on top of PyTorch, it provides a plug-and-play architecture where users can choose from a variety of RBM variants (Bernoulli, Gaussian, Convolutional, Discriminative, Dropout-based) and stack them into deep models with minimal code.

The library ships **18+ model variants** across four families: Bernoulli-based, Gaussian-based, extra activations, and deep architectures. All models inherit from `torch.nn.Module`, making them fully compatible with PyTorch's ecosystem for GPU acceleration, serialization, and gradient computation.

The core training algorithm is **Contrastive Divergence (CD-k)**, where the energy of the system is minimized through Gibbs sampling. Model quality is tracked via **Mean Squared Error (MSE)** for reconstruction and **log pseudo-likelihood (log-PL)** for density estimation.

---

## 2. High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          learnergy                                   │
├──────────┬──────────────┬──────────────┬──────────────┬──────────────┤
│   core   │    models    │     math     │    utils     │    visual    │
│ Dataset  │  bernoulli/  │   metrics    │  constants   │ convergence  │
│ Model    │  gaussian/   │   scale      │  exception   │ image        │
│          │  extra/      │              │  logging     │ tensor       │
│          │  deep/       │              │              │              │
└──────────┴──────────────┴──────────────┴──────────────┴──────────────┘
```

### Minimal Usage Pattern

```python
import torch
import torchvision
from learnergy.models.bernoulli import RBM

# Load dataset
train = torchvision.datasets.MNIST(root="./data", train=True, download=True,
                                    transform=torchvision.transforms.ToTensor())

# Create and train an RBM
model = RBM(n_visible=784, n_hidden=128, steps=1, learning_rate=0.1)
mse, pl = model.fit(train, batch_size=128, epochs=10)

# Reconstruct samples
mse, visible_probs = model.reconstruct(train)
```

---

## 3. Core Module (`learnergy/core/`)

The core module defines the two foundational abstractions upon which everything else is built.

### 3.1 Dataset (`dataset.py`)

A custom dataset class inheriting from `torch.utils.data.Dataset`. It wraps raw NumPy data and targets into a PyTorch-compatible dataset with optional transforms.

| Property | Type | Description |
|---|---|---|
| `data` | `np.array` | An n-dimensional array containing the input data |
| `targets` | `np.array` | A 1-dimensional array containing the labels |
| `transform` | `Callable` | Optional transform applied over each sample |

Key behavior:
- **`__getitem__(idx)`** — returns `(x, y)` tensor pairs, applying the transform to `x` if set
- **`__len__()`** — returns the dataset length
- **Validation** — the `transform` setter ensures the value is either callable or `None`

### 3.2 Model (`model.py`)

The base class for all learnable models, inheriting from `torch.nn.Module`. It provides device management and training history tracking.

| Property | Type | Description |
|---|---|---|
| `device` | `str` | Either `"cpu"` or `"cuda"` — auto-detected based on GPU availability and `use_gpu` flag |
| `history` | `Dict[str, Any]` | Dictionary of lists storing per-epoch metrics (MSE, log-PL, time, etc.) |

Key methods:
- **`dump(**kwargs)`** — appends any keyword arguments to their respective lists in `history`, creating new keys on-the-fly. This is called at the end of each training epoch to record metrics.

Design choice: the `Model` class sets `torch.set_default_tensor_type(torch.FloatTensor)` on initialization to ensure consistent floating-point precision across all operations.

---

## 4. Models Module (`learnergy/models/`)

The heart of the library — **18+ model variants** organized into four families. All models follow a consistent API: `fit()` for training, `reconstruct()` for reconstruction, and `forward()` for inference.

### 4.1 Bernoulli-Based (`bernoulli/`) — 6 models

Models where both visible and hidden units follow Bernoulli distributions, trained via Contrastive Divergence.

#### 4.1.1 RBM (`rbm.py`)

The foundational model. A Bernoulli-Bernoulli Restricted Boltzmann Machine with CD-k training.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_visible` | `int` | 128 | Number of visible units |
| `n_hidden` | `int` | 128 | Number of hidden units |
| `steps` | `int` | 1 | Number of Gibbs sampling steps (CD-k) |
| `learning_rate` | `float` | 0.1 | Learning rate |
| `momentum` | `float` | 0.0 | SGD momentum |
| `decay` | `float` | 0.0 | Weight decay (L2 regularization) |
| `temperature` | `float` | 1.0 | Temperature scaling factor for sampling |
| `use_gpu` | `bool` | False | GPU acceleration toggle |

Learnable parameters:
- **`W`** — Weight matrix of shape `(n_visible, n_hidden)`, initialized from `N(0, 0.01)`
- **`a`** — Visible bias of shape `(n_visible,)`, initialized to zeros
- **`b`** — Hidden bias of shape `(n_hidden,)`, initialized to zeros

Core methods:
- **`hidden_sampling(v)`** — computes `P(h|v) = σ(Wᵀv + b)` and samples binary states via `torch.bernoulli`
- **`visible_sampling(h)`** — computes `P(v|h) = σ(Wh + a)` and samples binary states
- **`gibbs_sampling(v)`** — runs the full CD-k procedure: positive phase → k steps of alternating visible/hidden sampling → negative phase
- **`energy(samples)`** — computes the free energy: `E = -vᵀa - Σ softplus(Wᵀv + b)`
- **`pseudo_likelihood(samples)`** — estimates log pseudo-likelihood by flipping random bits
- **`fit(dataset, batch_size, epochs)`** — trains the model, returning MSE and log-PL
- **`reconstruct(dataset)`** — one-step reconstruction, returning MSE and visible probabilities
- **`forward(x)`** — returns hidden probabilities (used for feature extraction or as input to downstream models)

Reference: G. Hinton. *A practical guide to training restricted Boltzmann machines.* Neural networks: Tricks of the trade (2012).

#### 4.1.2 ConvRBM (`conv_rbm.py`)

Convolutional variant using 2D convolutions (`F.conv2d` / `F.conv_transpose2d`) instead of linear projections. Supports optional MaxPooling2D.

| Additional Parameter | Type | Default | Description |
|---|---|---|---|
| `visible_shape` | `(int, int)` | (28, 28) | Spatial dimensions of the input |
| `filter_shape` | `(int, int)` | (7, 7) | Convolutional filter size |
| `n_filters` | `int` | 5 | Number of convolutional filters |
| `n_channels` | `int` | 1 | Number of input channels |
| `maxpooling` | `bool` | False | Whether to apply MaxPool2D in forward pass |
| `pooling_kernel` | `int` | 2 | Kernel size for MaxPool2D |

The hidden shape is automatically computed as `(visible - filter + 1, visible - filter + 1)`.

Reference: H. Lee et al. *Convolutional deep belief networks for scalable unsupervised learning of hierarchical representations.* ICML (2009).

#### 4.1.3 DiscriminativeRBM (`discriminative_rbm.py`)

Extends RBM for supervised classification. Adds class-specific weights `U` of shape `(n_classes, n_hidden)` and class bias `c`. Uses cross-entropy loss instead of the generative energy objective.

Key methods:
- **`labels_sampling(samples)`** — computes `P(y|v)` via softplus-based logit probabilities
- **`fit()`** — trains using cross-entropy loss, returns loss and accuracy
- **`predict(dataset)`** — returns accuracy, probabilities, and predicted labels

Reference: H. Larochelle and Y. Bengio. *Classification using discriminative restricted Boltzmann machines.* ICML (2008).

#### 4.1.4 HybridDiscriminativeRBM (`discriminative_rbm.py`)

Extends `DiscriminativeRBM` with a hybrid loss: `L = L_discriminative + α × L_generative`. Adds class layer sampling `P(y|h)` and a joint Gibbs sampling procedure over `(v, y, h)`.

| Additional Parameter | Type | Default | Description |
|---|---|---|---|
| `alpha` | `float` | 0.01 | Weight of the generative loss component |

#### 4.1.5 DropoutRBM (`dropout_rbm.py`)

Applies standard Dropout regularization to hidden unit activations during training. A Bernoulli mask with probability `(1 - p)` is element-wise multiplied with hidden probabilities.

| Additional Parameter | Type | Default | Description |
|---|---|---|---|
| `dropout` | `float` | 0.5 | Dropout probability |

During reconstruction, dropout is temporarily disabled (`p = 0`) to use the full network capacity.

Reference: N. Srivastava et al. *Dropout: a simple way to prevent neural networks from overfitting.* JMLR (2014).

#### 4.1.6 DropConnectRBM (`dropout_rbm.py`)

Variant of `DropoutRBM` that applies the Bernoulli mask to the **weight matrix** rather than the hidden activations (`W ⊙ mask` instead of `h ⊙ mask`).

### 4.2 Gaussian-Based (`gaussian/`) — 7 models

Models where the visible layer uses Gaussian units, suitable for continuous-valued data (e.g., real-valued pixel intensities).

#### 4.2.1 GaussianRBM (`gaussian_rbm.py`)

Gaussian-Bernoulli RBM with standardized inputs (variance fixed to 1). The energy function changes to include a quadratic visible term: `E = 0.5 × Σ(v - a)² - Σ softplus(Wᵀv + b)`.

| Additional Parameter | Type | Default | Description |
|---|---|---|---|
| `normalize` | `bool` | True | Apply batch-wise zero-mean, unit-variance normalization |
| `input_normalize` | `bool` | True | Normalize inputs during forward pass |

The visible sampling returns raw activations (linear units) instead of Bernoulli samples, reflecting the Gaussian assumption.

Reference: K. Cho, A. Ilin, T. Raiko. *Improved learning of Gaussian-Bernoulli restricted Boltzmann machines.* ICANN (2011).

#### 4.2.2 GaussianReluRBM (`gaussian_rbm.py`)

Replaces sigmoid activation in the hidden layer with **ReLU** (`F.relu`), allowing for sparse, non-negative hidden representations. Designed for modeling image covariance from raw integer pixel values.

#### 4.2.3 GaussianSeluRBM (`gaussian_rbm.py`)

Replaces sigmoid activation with **SeLU** (`F.selu`) for self-normalizing behavior in deeper architectures.

Reference: G. Klambauer et al. *Self-normalizing neural networks.* NeurIPS (2017).

#### 4.2.4 VarianceGaussianRBM (`gaussian_rbm.py`)

Gaussian-Bernoulli RBM **without** input standardization. Instead, learns an explicit per-unit variance parameter `σ` (sigma). The hidden sampling divides inputs by `σ²`, and the energy incorporates the learned variance.

This allows the model to handle non-standardized data at the cost of an additional learnable parameter.

#### 4.2.5 GaussianConvRBM (`gaussian_conv_rbm.py`)

Gaussian-based convolutional RBM using **ReLU6** activation in the hidden layer and linear (identity) activations in the visible layer when normalization is enabled. Batch normalization is applied to input samples.

#### 4.2.6 GaussianRBM4deep / GaussianReluRBM4deep (`gaussian_rbm.py`)

Specialized variants of `GaussianRBM` and `GaussianReluRBM` designed for use as inner layers of a DBN. They suppress per-epoch logging to avoid noise during the greedy layer-wise training procedure.

#### 4.2.7 GaussianConvRBM4Deep (`gaussian_conv_rbm.py`)

Convolutional Gaussian RBM variant for deeper layers in a Convolutional DBN. Provides a `log` flag to control training verbosity.

### 4.3 Extra Models (`extra/`) — 2 models

#### 4.3.1 SigmoidRBM (`sigmoid_rbm.py`)

A Sigmoid-Bernoulli RBM where the visible sampling returns **probabilities directly as states** (deterministic visible layer), rather than sampling from Bernoulli. This produces smoother reconstructions.

#### 4.3.2 SigmoidRBM4Deep (`sigmoid_rbm.py`)

Variant of `SigmoidRBM` designed for inner layers of a DBN. Suppresses per-epoch logging during greedy layer-wise training.

### 4.4 Deep Models (`deep/`) — 3 models

Deep architectures formed by stacking multiple RBMs and training them via greedy layer-wise pre-training.

#### 4.4.1 DBN (`dbn.py`)

A Deep Belief Network composed of a stack of RBMs. The first layer can be any model type; subsequent layers automatically fall back to their `4deep` variants (e.g., `"gaussian"` → `"gaussian4deep"`, `"sigmoid"` → `"sigmoid4deep"`).

| Parameter | Type | Description |
|---|---|---|
| `model` | `Tuple[str, ...]` | RBM type per layer. Supported keys: `"bernoulli"`, `"dropout"`, `"e_dropout"`, `"gaussian"`, `"gaussian_relu"`, `"gaussian_selu"`, `"sigmoid"`, `"variance_gaussian"` |
| `n_visible` | `int` | Input dimensionality |
| `n_hidden` | `Tuple[int, ...]` | Hidden units per layer — defines the number of layers |
| `steps`, `learning_rate`, `momentum`, `decay`, `temperature` | `Tuple[...]` | Per-layer hyperparameters |

Training procedure (greedy layer-wise):
1. **Layer 0** is trained on the original dataset using its `fit()` method
2. **Layer i > 0** is trained on the hidden representations of layer `i-1`, reconstructed batch-by-batch via `hidden_sampling()` of all preceding layers

The model registry `MODELS` maps string keys to RBM classes:
```python
MODELS = {
    "bernoulli": RBM,
    "dropout": DropoutRBM,
    "e_dropout": EDropoutRBM,
    "gaussian": GaussianRBM,
    "gaussian4deep": GaussianRBM4deep,
    "gaussian_relu": GaussianReluRBM,
    "gaussian_relu4deep": GaussianReluRBM4deep,
    "gaussian_selu": GaussianSeluRBM,
    "sigmoid": SigmoidRBM,
    "sigmoid4deep": SigmoidRBM4Deep,
    "variance_gaussian": VarianceGaussianRBM,
}
```

Reference: G. Hinton, S. Osindero, Y. Teh. *A fast learning algorithm for deep belief nets.* Neural Computation (2006).

#### 4.4.2 ConvDBN (`conv_dbn.py`)

A Convolutional Deep Belief Network stacking `ConvRBM` or `GaussianConvRBM` layers. Visible shapes are automatically propagated between layers based on filter sizes and optional max-pooling.

| Additional Parameter | Type | Description |
|---|---|---|
| `visible_shape` | `(int, int)` | Spatial input shape |
| `filter_shape` | `Tuple[(int, int), ...]` | Filter shape per layer |
| `n_filters` | `Tuple[int, ...]` | Number of filters per layer |
| `maxpooling` | `Tuple[bool, ...]` | Whether to pool after each layer |
| `pooling_kernel` | `Tuple[int, ...]` | Pooling kernel size per layer |

Reference: H. Lee et al. *Convolutional deep belief networks for scalable unsupervised learning of hierarchical representations.* ICML (2009).

#### 4.4.3 ResidualDBN (`residual_dbn.py`)

A DBN with **residual (skip) connections** between layers. After each layer's forward pass, the output is combined with a ReLU-normalized version of the pre-activation:

```
output = ζ₁ × hidden + ζ₂ × ReLU_normalized(pre_activation)
```

| Additional Parameter | Type | Default | Description |
|---|---|---|---|
| `zetta1` | `float` | 1.0 | Weight for the original hidden representation |
| `zetta2` | `float` | 1.0 | Weight for the residual component |

Reference: M. Roder et al. *A Layer-Wise Information Reinforcement Approach to Improve Learning in Deep Belief Networks.* ICAISC (2020).

---

## 5. Math Module (`learnergy/math/`)

Low-level mathematical utilities used across the library.

### 5.1 `metrics.py` — Quality Metrics

- **`calculate_ssim(v, x)`** — Computes the Structural Similarity Index (SSIM) between reconstructed images `v` and original images `x`. Converts tensors to NumPy and uses `skimage.metrics.structural_similarity`. Returns the average SSIM across all samples in the batch.

### 5.2 `scale.py` — Data Scaling

- **`unitary_scale(x)`** — Scales a NumPy array to the range `[0, 1]` using min-max normalization. Adds `EPSILON` to the denominator to prevent division by zero.

---

## 6. Utils Module (`learnergy/utils/`)

### 6.1 `constants.py` — Global Constants

| Constant | Value | Purpose |
|---|---|---|
| `EPSILON` | `1e-10` | Prevents division by zero, zero logarithms, and numerical instability |

### 6.2 `exception.py` — Custom Exceptions

Five exception types, all inheriting from a base `Error` class that logs to the logger:

| Exception | Purpose |
|---|---|
| `ArgumentError` | Wrong number of provided arguments |
| `BuildError` | Class not properly built before use |
| `SizeError` | Mismatched array/tuple sizes (e.g., per-layer hyperparameters) |
| `TypeError` | Wrong variable type (e.g., non-callable transform) |
| `ValueError` | Out-of-range values (e.g., negative learning rate) |

### 6.3 `logging.py` — Logging Infrastructure

Custom logging setup with dual output:
- **Console handler** — `StreamHandler(sys.stdout)` for immediate feedback
- **Timed file handler** — `TimedRotatingFileHandler` writing to `learnergy.log` with daily rotation

Format: `%(asctime)s - %(name)s — %(levelname)s — %(message)s`

All loggers are created via `get_logger(name)` and set to `DEBUG` level with `propagate=False` to prevent duplicate messages.

---

## 7. Visualization Module (`learnergy/visual/`)

### 7.1 `convergence.py` — Training Convergence

- **`plot(*args, labels, title, ...)`** — Plots 2D convergence curves (epoch × value) for one or more variables using Matplotlib. Supports customizable titles, axis labels, grid, and legend.

### 7.2 `image.py` — Image Visualization

- **`_rasterize(x, img_shape, tile_shape, ...)`** — Internal function that arranges weight vectors or samples into a tiled mosaic grid, with optional unitary scaling
- **`create_mosaic(tensor)`** — Creates and displays a weight mosaic from a 2D tensor (e.g., RBM weights), using Pillow
- **`create_rgb_mosaic(tensor, n_samples)`** — Displays a grid of RGB images from a 4D tensor `(batch, channels, height, width)`

### 7.3 `tensor.py` — Tensor Visualization

- **`save_tensor(tensor, output_path)`** — Saves a tensor as a grayscale (or RGB if 3-channel) image to disk
- **`show_tensor(tensor)`** — Displays a tensor as a grayscale (or RGB) image using Matplotlib

---

## 8. The Training Loop

All RBM-based models follow a consistent training loop within their `fit()` method:

```
FOR epoch = 1..epochs:
  1. Create DataLoader from dataset
  2. FOR each batch:
     a. Reshape input to match model dimensions
     b. Move to GPU if enabled
     c. [Optional] Normalize batch (Gaussian models)
     d. gibbs_sampling(v)          # CD-k: positive → negative phase
     e. Compute energy-based cost: E(data) - E(reconstruction)
     f. optimizer.zero_grad()
     g. cost.backward()            # Backprop through energy
     h. optimizer.step()           # Update W, a, b via SGD
     i. Compute batch MSE and log-PL
  3. Average metrics across batches
  4. model.dump(mse=..., pl=..., time=...)  # Record to history
```

The energy-based cost function is the difference between the mean energy of the data and the mean energy of the reconstructed samples. This is equivalent to the gradient of the log-likelihood under the Contrastive Divergence approximation.

---

## 9. Class Hierarchy

```
torch.nn.Module
  └── Model (core/model.py)
        ├── RBM (bernoulli/rbm.py)
        │     ├── DropoutRBM (bernoulli/dropout_rbm.py)
        │     │     └── DropConnectRBM (bernoulli/dropout_rbm.py)
        │     ├── EDropoutRBM (bernoulli/e_dropout_rbm.py)
        │     ├── DiscriminativeRBM (bernoulli/discriminative_rbm.py)
        │     │     └── HybridDiscriminativeRBM (bernoulli/discriminative_rbm.py)
        │     ├── SigmoidRBM (extra/sigmoid_rbm.py)
        │     │     └── SigmoidRBM4Deep (extra/sigmoid_rbm.py)
        │     ├── GaussianRBM (gaussian/gaussian_rbm.py)
        │     │     ├── GaussianReluRBM (gaussian/gaussian_rbm.py)
        │     │     ├── GaussianSeluRBM (gaussian/gaussian_rbm.py)
        │     │     └── GaussianRBM4deep (gaussian/gaussian_rbm.py)
        │     │           └── GaussianReluRBM4deep (gaussian/gaussian_rbm.py)
        │     └── VarianceGaussianRBM (gaussian/gaussian_rbm.py)
        ├── ConvRBM (bernoulli/conv_rbm.py)
        │     ├── GaussianConvRBM (gaussian/gaussian_conv_rbm.py)
        │     └── GaussianConvRBM4Deep (gaussian/gaussian_conv_rbm.py)
        ├── DBN (deep/dbn.py)
        │     └── ResidualDBN (deep/residual_dbn.py)
        └── ConvDBN (deep/conv_dbn.py)

torch.utils.data.Dataset
  └── Dataset (core/dataset.py)
```

---

## 10. Design Patterns & Conventions

### 10.1 Property Validation

Every class uses Python properties with setters that perform **strict type and value validation**, raising custom exceptions (`learnergy.utils.exception`). This defensive style catches configuration errors early (e.g., negative learning rates, zero-sized layers) rather than producing cryptic tensor errors during training.

```python
@n_visible.setter
def n_visible(self, n_visible: int) -> None:
    if n_visible <= 0:
        raise e.ValueError("`n_visible` should be > 0")
    self._n_visible = n_visible
```

### 10.2 Consistent Model API

All models expose the same public interface:

| Method | Returns | Description |
|---|---|---|
| `fit(dataset, batch_size, epochs)` | `(mse, pl)` or `mse` | Train the model |
| `reconstruct(dataset)` | `(mse, visible_probs)` | One-step reconstruction |
| `forward(x)` | `tensor` | Hidden layer activation (feature extraction) |

Deep models (`DBN`, `ConvDBN`) return lists of per-layer metrics.

### 10.3 Template Method Pattern

The base `RBM` class defines the skeleton (energy computation, Gibbs sampling, training loop), and subclasses override specific methods:
- **`hidden_sampling()`** — customized by `DropoutRBM` (masking), `GaussianReluRBM` (ReLU), `GaussianSeluRBM` (SeLU)
- **`visible_sampling()`** — customized by `GaussianRBM` (linear units), `SigmoidRBM` (deterministic)
- **`energy()`** — customized by `GaussianRBM` (quadratic visible term), `VarianceGaussianRBM` (learned variance)
- **`fit()`** — customized by `EDropoutRBM` (energy-based masking), `DiscriminativeRBM` (cross-entropy)

### 10.4 Model Registry Pattern

Deep models use a dictionary-based registry (`MODELS`) to map string identifiers to RBM classes. This enables dynamic model composition through configuration:

```python
dbn = DBN(model=("gaussian", "sigmoid", "bernoulli"),
          n_hidden=(256, 128, 64), ...)
```

### 10.5 Greedy Layer-Wise Training

DBNs are trained layer by layer: each RBM is trained on the hidden representations produced by all preceding layers. This follows the original Hinton et al. (2006) procedure and is implemented by iterating through `self.models` and passing transformed data forward.

### 10.6 History Tracking via `dump()`

The `Model.dump(**kwargs)` method provides a flexible, schema-free history system. Any metric can be recorded by simply passing it as a keyword argument — the history dictionary grows dynamically. This is used across all models to track `mse`, `pl`, `loss`, `acc`, and `time`.

### 10.7 PyTorch Integration

All models are `torch.nn.Module` subclasses, which means:
- Learnable parameters are registered as `nn.Parameter` and automatically tracked
- `model.cuda()` moves all parameters to GPU
- `model.parameters()` works with any PyTorch optimizer
- Models can be serialized with `torch.save()` / `torch.load()`
- The `forward()` method enables models to be used as feature extractors in larger PyTorch pipelines

---

## 11. Dependencies

| Package | Version | Purpose |
|---|---|---|
| `torch` | ≥ 1.8.0 | Core tensor operations, neural network modules, GPU support |
| `torchvision` | ≥ 0.9.0 | Dataset loading (MNIST, CIFAR, etc.) and image transforms |
| `matplotlib` | ≥ 3.3.4 | Convergence plots and tensor visualization |
| `Pillow` | ≥ 8.1.2 | Image mosaic creation and display |
| `scikit-image` | ≥ 0.17.2 | SSIM metric computation |
| `tqdm` | ≥ 4.49.0 | Training progress bars |
| `requests` | ≥ 2.23.0 | HTTP utilities |

`pre-commit` is available as an optional development dependency via `pip install learnergy[dev]`.

---

## 12. Directory Structure

```
learnergy/
├── __init__.py              # Package version (1.1.5)
├── core/
│   ├── dataset.py           # Dataset wrapper (torch.utils.data.Dataset)
│   └── model.py             # Base Model class (torch.nn.Module)
├── math/
│   ├── metrics.py           # SSIM calculation
│   └── scale.py             # Min-max scaling
├── models/
│   ├── bernoulli/
│   │   ├── rbm.py           # RBM (base)
│   │   ├── conv_rbm.py      # ConvRBM
│   │   ├── discriminative_rbm.py  # DiscriminativeRBM, HybridDiscriminativeRBM
│   │   ├── dropout_rbm.py   # DropoutRBM, DropConnectRBM
│   │   └── e_dropout_rbm.py # EDropoutRBM
│   ├── deep/
│   │   ├── dbn.py           # DBN
│   │   ├── conv_dbn.py      # ConvDBN
│   │   └── residual_dbn.py  # ResidualDBN
│   ├── extra/
│   │   └── sigmoid_rbm.py   # SigmoidRBM, SigmoidRBM4Deep
│   └── gaussian/
│       ├── gaussian_rbm.py  # GaussianRBM + variants
│       └── gaussian_conv_rbm.py  # GaussianConvRBM + variants
├── utils/
│   ├── constants.py         # EPSILON
│   ├── exception.py         # Custom exception hierarchy
│   └── logging.py           # Dual-output logging setup
└── visual/
    ├── convergence.py       # Convergence plots
    ├── image.py             # Mosaic creation
    └── tensor.py            # Tensor save/show
```

---

## 13. Numerical Stability Design Decisions

The following defensive patterns are applied throughout the codebase to prevent silent NaN/Inf propagation and ensure reliable training:

### 13.1 Epsilon Guards on Division

All divisions where the denominator may be zero are guarded with `c.EPSILON` (= `1e-10`):

| Location | Risk | Fix |
|---|---|---|
| `EDropoutRBM.energy_dropout()` | Dropout probabilities can be zero | `/ (p + c.EPSILON)` |
| `ResidualDBN.calculate_residual()` | `torch.max(x)` is zero for all-zero tensors after ReLU | `/ (max_val + c.EPSILON)` |
| `ResidualDBN.fit()` / `forward()` | Same ReLU zero-output risk | `/ (max_val + c.EPSILON)` |
| `GaussianRBM.hidden_sampling()` | Sigma² can collapse to zero | `sigma.clamp(min=c.EPSILON)` |
| `GaussianRBM.energy()` | Same sigma² denominator risk | `sigma.clamp(min=c.EPSILON)` |
| `math/scale.py` | `max - min` is zero for constant arrays | `/ (range + c.EPSILON)` |

### 13.2 Correct Per-Sample Energy

Energy functions must return per-sample scalars (shape `[batch_size]`), not cross-batch matrices. The corrected patterns are:

- **EDropoutRBM.total_energy()**: Uses `torch.sum((v @ W) * h, dim=1)` instead of `torch.mm(v, (W @ h.t()))`, which produced a `(batch, batch)` matrix
- **ConvRBM.energy()**: Uses per-channel bias via `self.a.view(1, -1, 1, 1)` instead of `torch.mean(self.a)`, which averaged biases across all channels

### 13.3 Detached Pseudo-Likelihood

The `pseudo_likelihood()` method in `EDropoutRBM` applies `.detach()` to the corrupted round-trip tensor before computing the log-likelihood. This prevents gradients from flowing through the diagnostic metric and avoids interfering with the CD-k training objective.

### 13.4 Exception Hierarchy

Custom exceptions (`TypeError`, `ValueError`) now inherit from **both** the learnergy `Error` base class and the corresponding Python builtin exception. This ensures they are caught by standard `except TypeError:` / `except ValueError:` clauses while retaining the library's logging behavior:

```python
class TypeError(Error, builtins.TypeError):
    ...
```

### 13.5 Specific Exception Handling

Bare `except:` clauses are replaced with specific `except (AttributeError, TypeError):` to avoid swallowing unexpected errors (e.g., `KeyboardInterrupt`, `SystemExit`). This applies to the dataset conversion logic in `DBN`, `ConvDBN`, and related deep models.

### 13.6 Logger Handler Deduplication

The `get_logger()` function guards against handler accumulation with `if not logger.handlers:` before adding console and file handlers. Without this, each call would add duplicate handlers, causing log messages to repeat exponentially.

### 13.7 Temperature Constraint

The base `RBM` allows any temperature `T > 0` (not just `T ≥ 1`), enabling simulated annealing and temperature-based exploration during training.

### 13.8 Reconstruction Normalization

`RBM.reconstruct()` normalizes MSE by `samples.size(0)` (the number of test samples) rather than `dataset.size(0)` (the full dataset), ensuring correct per-sample error regardless of batch boundaries.

### 13.9 Deprecated API Migration

| Deprecated API | Replacement | Affected File |
|---|---|---|
| `torch.set_default_tensor_type()` | `torch.set_default_dtype()` | `core/model.py` |
| `plt.cm.get_cmap("gray")` | `"gray"` string literal | `visual/tensor.py` |
| `np.array` in type hints | `np.ndarray` | `core/dataset.py`, `math/scale.py` |

### 13.10 Resource Cleanup

Matplotlib figure handles are explicitly closed with `plt.close(fig)` after rendering to prevent memory leaks during batch visualization.
