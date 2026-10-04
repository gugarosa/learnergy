Temporal models
===============

The ``learnergy.models.temporal`` family contains ``RTRBM``, ``RTGaussianRBM``,
``RTVarianceGaussianRBM``, and ``RTDBN``. They extend energy-based learning to
sequences without adding runtime dependencies.

Sequence contracts
------------------

Inputs are finite floating-point tensors shaped
``(batch_size, sequence_length, n_visible)``. All dimensions must be nonempty.
Dataset items contain a sequence shaped ``(sequence_length, n_visible)`` and a
target. Single-layer training ignores targets, while stacked training preserves
them when materializing the next layer's dataset.

Each sequence starts from the learned ``h0`` context. The next hidden bias
depends on the preceding hidden probabilities through ``W_prime``, not on
sampled hidden states. Context does not persist between calls.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Operation
     - Result
   * - Temporal RBM ``model(sequences)``
     - Hidden probabilities shaped ``(batch_size, sequence_length, n_hidden)``.
   * - Temporal RBM ``fit(dataset)``
     - Final-epoch detached scalar reconstruction error.
   * - Temporal RBM ``reconstruct(dataset)``
     - Detached scalar error followed by reconstructed sequence values.
   * - ``sample(n_samples, n_steps, gibbs_steps)``
     - Detached visible states shaped ``(n_samples, n_steps, n_visible)``.
   * - ``RTDBN.encode(sequences)`` or ``model(sequences)``
     - Mean-pooled embeddings shaped ``(batch_size, n_hidden[-1])``.
   * - ``RTDBN.fit(dataset)``
     - Final-epoch MSE floats in layer order.

Sequences within a batch must have the same length. Padding masks,
variable-length collation, persistent streaming context, and truncated
backpropagation are outside this interface.

Training and generation
-----------------------

.. code-block:: python

   import torch
   from torch.utils.data import TensorDataset

   from learnergy.models.temporal import RTRBM

   sequences = torch.bernoulli(torch.rand(32, 6, 4))
   dataset = TensorDataset(sequences, torch.zeros(32))
   model = RTRBM(n_visible=4, n_hidden=8, learning_rate=0.01)

   mse = model.fit(dataset, batch_size=8, epochs=2)
   hidden_sequences = model(sequences)
   reconstruction_mse, reconstructed = model.reconstruct(dataset)
   generated = model.sample(n_samples=3, n_steps=6, gibbs_steps=20)

``fit_subseries`` updates a complete sequence batch with one backward pass.
Observed inputs and hidden probabilities retain their recurrent graph; negative
Gibbs particles are detached. ``cd_step`` instead performs one parameter update
for a single timestep with an explicitly supplied context.

Batch reconstruction error is the squared error summed over features and time,
then averaged over sequences. Epoch MSE is the mean of these batch errors,
including the final partial batch. ``fit`` appends MSE and elapsed-time floats to
the model's existing history. ``fit_subseries`` and ``cd_step`` do not append
history.

Reconstruction uses observed inputs to advance recurrent context and processes
the dataset in one non-shuffled batch. It returns Bernoulli probabilities or
Gaussian conditional means, retaining model gradients in those values. Error
is calculated against visible states, which are stochastic for Bernoulli and
learned-variance Gaussian models.

Generation restarts a Gibbs chain at each timestep and updates recurrent
context from generated observations. Sampling does not track gradients,
modify parameters, append history, or change training mode. Counts must be
positive integers.

The self-contained example can be run from the repository root:

.. code-block:: console

   uv run python -m examples.applications.temporal.rtrbm_training

Gaussian behavior
-----------------

``RTGaussianRBM`` uses unit visible variance. Its normalization flags follow
``GaussianRBM``:

* ``normalize`` controls training and reconstruction standardization.
* ``input_normalize`` controls standardization in ``forward``.

Standardization pools batch and time dimensions separately for each feature.
More than one observation uses sample standard deviation; a singleton is
centered to zero. Standardized inputs are detached. No training statistics are
stored, and outputs are not transformed back to the original units. Disable
normalization when supplying externally standardized features or when input
gradients must pass through a fixed-variance Gaussian layer.

As in ``GaussianRBM``, ``RTGaussianRBM.visible_sampling`` returns sigmoid values
followed by deterministic continuous means. Contrastive Divergence and
reconstruction use the continuous means, not the sigmoid values. Generative
sampling adds unit Gaussian noise, including when this model is a lower RTDBN
layer.

``RTVarianceGaussianRBM`` does not standardize its inputs. Its effective
variance is ``sigma**2`` plus the input dtype's machine epsilon. Hidden
conditionals and free energy use that same variance. The Gaussian quadratic
contribution to free energy is positive.

As in ``VarianceGaussianRBM``, ``visible_sampling`` returns conditional means
followed by random visible states. Sampling uses the square root of the
effective variance as its standard deviation, and Contrastive Divergence uses
the random states as negative particles. The visible distribution is unchanged
by the sampler's compatibility ``scale`` argument.

Both temporal Gaussian variants clip the total training gradient norm to one.
Learned scales are bounded to ``[0.1, 10]`` after an update only when they are
trainable; explicitly frozen scales remain unchanged. Nonfinite input
sequences and nonfinite clipped gradients raise errors rather than being
replaced with plausible values.

Stacked temporal models
-----------------------

Each RTDBN layer accepts ``bernoulli``, ``gaussian``, or
``variance_gaussian``. Model names and constructor hyperparameter tuples must
contain one entry per layer. The default remains a single learned-variance
Gaussian layer.

.. code-block:: python

   from learnergy.models.temporal import RTDBN

   dataset = TensorDataset(torch.randn(32, 6, 4), torch.arange(32))
   model = RTDBN(
       model=("gaussian", "bernoulli"),
       n_visible=4,
       n_hidden=(3, 2),
       steps=(1, 1),
       learning_rate=(0.001, 0.001),
       momentum=(0.0, 0.0),
       decay=(0.0, 0.0),
       temperature=(1.0, 1.0),
       normalize=False,
       input_normalize=False,
   )

   errors = model.fit(dataset, batch_size=8, epochs=(2, 2), warmup_epochs=())
   embeddings = model.encode(dataset.tensors[0])
   generated = model.sample(n_samples=3, n_steps=6, gibbs_steps=20)

Training is greedy and layer-wise. Before fitting the next layer, the previous
layer's sequence probabilities and original targets are collected into a
detached CPU ``TensorDataset``. Earlier layers are not updated by later-layer
training. Existing parameter freezes and module training modes are preserved.
Encoding calls each layer through PyTorch's module interface, including hooks.

``epochs`` specifies each layer's total epoch budget, including scale warmup.
``warmup_epochs`` temporarily freezes ``sigma`` only for learned-variance
layers. Warmup is capped at the corresponding total epoch count, omitted
entries mean zero, and an empty tuple disables warmup. The previous scale
gradient flag is restored even when training raises an exception.

Stacked generation runs recurrent Gibbs sampling at the top layer, then samples
each lower visible conditional once per timestep. Only encoding mean-pools the
time dimension; training features and generated sequences retain it.

Devices, gradients, and checkpoints
-----------------------------------

``fit`` and ``reconstruct`` move dataset tensors to the model's current device
and dtype. Direct tensor operations expect inputs on that device and dtype.
Generated tensors follow model parameters, including after ``double()`` or
device movement.

Forward operations preserve gradients through the recurrent computation unless
fixed-variance Gaussian normalization detaches an input. Use
``torch.no_grad()`` when extracting frozen features. ``pseudo_likelihood``
retains the Bernoulli bit-flip estimator with fixed recurrent context; it is not
a continuous Gaussian likelihood.

Checkpoint parameter names remain ``W``, ``a``, ``b``, ``W_prime``, and ``h0``,
with ``sigma`` for learned variance. RTDBN prefixes these names with
``models.<layer>.``. Model imports and existing constructor defaults are
preserved. Corrected numerical semantics can change training and generation
even when a checkpoint or random seed is unchanged.
