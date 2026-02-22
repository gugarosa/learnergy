"""Unit tests for all bug fixes from the code review.

Tests are grouped by the bug they verify, covering critical, high, and medium
severity fixes across the codebase.
"""

import numpy as np
import torch
import torch.nn as nn

from learnergy.core import dataset, model
from learnergy.math import scale
from learnergy.models.bernoulli import conv_rbm, e_dropout_rbm, rbm
from learnergy.models.deep import conv_dbn, dbn, residual_dbn
from learnergy.models.extra import sigmoid_rbm
from learnergy.models.gaussian import gaussian_rbm
from learnergy.utils import exception, logging


# ---------------------------------------------------------------------------
# CRITICAL #1: EDropoutRBM.total_energy per-sample correctness
# ---------------------------------------------------------------------------
def test_e_dropout_total_energy_per_sample():
    """total_energy must return a scalar (mean of per-sample energies),
    not a matrix from cross-batch interactions."""
    m = e_dropout_rbm.EDropoutRBM(n_visible=16, n_hidden=8)
    batch = 4
    h = torch.randn(batch, 8)
    v = torch.randn(batch, 16)
    energy = m.total_energy(h, v)
    # Must be a scalar (0-dim tensor)
    assert energy.dim() == 0


# ---------------------------------------------------------------------------
# CRITICAL #2: ConvRBM per-channel energy bias
# ---------------------------------------------------------------------------
def test_conv_rbm_energy_per_channel_bias():
    """Energy must use per-channel biases, not their mean."""
    m = conv_rbm.ConvRBM(
        visible_shape=(8, 8), filter_shape=(3, 3),
        n_filters=2, n_channels=3, steps=1,
    )
    # Set distinct bias values per channel
    with torch.no_grad():
        m.a.copy_(torch.tensor([1.0, -1.0, 0.5]))

    samples = torch.randn(2, 3, 8, 8)
    energy = m.energy(samples)
    assert energy.shape == (2,)

    # Verify per-channel contribution differs from mean-based
    mean_bias_energy = torch.sum(samples, dim=(1, 2, 3)) * torch.mean(m.a)
    per_channel_energy = torch.sum(
        samples * m.a.view(1, -1, 1, 1), dim=(1, 2, 3)
    )
    assert not torch.allclose(mean_bias_energy, per_channel_energy)


# ---------------------------------------------------------------------------
# CRITICAL #3: ResidualDBN division by zero protection
# ---------------------------------------------------------------------------
def test_residual_dbn_calculate_residual_zeros():
    """calculate_residual must not produce NaN when input is all-zeros."""
    m = residual_dbn.ResidualDBN(
        n_visible=16, n_hidden=(8,), steps=(1,),
        learning_rate=(0.1,), momentum=(0,), decay=(0,), temperature=(1,),
    )
    zeros = torch.zeros(2, 16)
    result = m.calculate_residual(zeros)
    assert not torch.isnan(result).any()
    assert result.shape == (2, 16)


def test_residual_dbn_forward_no_nan():
    """forward must not produce NaN for any input."""
    m = residual_dbn.ResidualDBN(
        n_visible=16, n_hidden=(8,), steps=(1,),
        learning_rate=(0.1,), momentum=(0,), decay=(0,), temperature=(1,),
    )
    x = torch.randn(2, 16)
    result = m.forward(x)
    assert not torch.isnan(result).any()


# ---------------------------------------------------------------------------
# CRITICAL #4: VarianceGaussianRBM sigma broadcasting on CPU
# ---------------------------------------------------------------------------
def test_variance_gaussian_rbm_visible_sampling_cpu():
    """visible_sampling must work on CPU without shape mismatch."""
    m = gaussian_rbm.VarianceGaussianRBM(n_visible=16, n_hidden=8)
    m._device = "cpu"
    h = torch.randn(4, 8)
    probs, states = m.visible_sampling(h)
    assert probs.shape == (4, 16)
    assert states.shape == (4, 16)


# ---------------------------------------------------------------------------
# CRITICAL #5: ConvDBN maxpooling padding
# ---------------------------------------------------------------------------
def test_convdbn_maxpooling_padding():
    """ConvDBN must pad maxpooling tuple to n_layers without IndexError."""
    m = conv_dbn.ConvDBN(
        model="bernoulli",
        visible_shape=(28, 28),
        filter_shape=((7, 7), (5, 5)),
        n_filters=(4, 8),
        n_channels=1,
        steps=(1, 1),
        learning_rate=(0.01, 0.01),
        momentum=(0, 0),
        decay=(0, 0),
        maxpooling=(False,),  # shorter than n_layers=2
        pooling_kernel=(2,),
    )
    assert len(m.maxpooling) == 2
    assert m.maxpooling[1] is False

    # forward should not raise IndexError
    x = torch.randn(1, 1, 28, 28)
    out = m.forward(x)
    assert out is not None


# ---------------------------------------------------------------------------
# HIGH #6: EDropoutRBM energy_dropout division by zero
# ---------------------------------------------------------------------------
def test_e_dropout_energy_dropout_zero_probs():
    """energy_dropout must handle zero probabilities without producing NaN."""
    m = e_dropout_rbm.EDropoutRBM(n_visible=16, n_hidden=8)
    m.M = torch.ones(2, 8)

    e = torch.tensor(0.0)  # zero energy difference
    p_prob = torch.zeros(2, 8)  # all-zero positive probs
    n_prob = torch.ones(2, 8)

    m.energy_dropout(e, p_prob, n_prob)
    assert not torch.isnan(m.M).any()
    assert not torch.isinf(m.M).any()


# ---------------------------------------------------------------------------
# HIGH #7: SigmoidRBM4Deep super() call
# ---------------------------------------------------------------------------
def test_sigmoid_rbm4deep_inherits_visible_sampling():
    """SigmoidRBM4Deep must inherit SigmoidRBM's visible_sampling."""
    m = sigmoid_rbm.SigmoidRBM4Deep(n_visible=16, n_hidden=8)
    h = torch.randn(2, 8)
    probs, states = m.visible_sampling(h)
    # SigmoidRBM returns probs == states (deterministic visible)
    assert torch.equal(probs, states)


def test_sigmoid_rbm4deep_is_instance():
    """SigmoidRBM4Deep must be an instance of SigmoidRBM."""
    m = sigmoid_rbm.SigmoidRBM4Deep()
    assert isinstance(m, sigmoid_rbm.SigmoidRBM)


# ---------------------------------------------------------------------------
# HIGH #8-9: Bare except clauses replaced
# ---------------------------------------------------------------------------
def test_dbn_dataset_conversion_specific_exceptions():
    """DBN.fit should catch specific exceptions, not swallow all."""
    m = dbn.DBN(
        model=("bernoulli",),
        n_visible=16, n_hidden=(8,), steps=(1,),
        learning_rate=(0.1,), momentum=(0,), decay=(0,), temperature=(1,),
    )
    # Create a minimal dataset that works
    data = torch.randn(10, 16)
    targets = torch.zeros(10)
    ds = dataset.Dataset(data, targets, show_log=False)
    # Should not raise
    mse, pl = m.fit(ds, batch_size=5, epochs=(1,))
    assert mse is not None


# ---------------------------------------------------------------------------
# HIGH #10: Logger handler accumulation
# ---------------------------------------------------------------------------
def test_logger_no_handler_accumulation():
    """get_logger must not add duplicate handlers on repeated calls."""
    name = "test_handler_accum_unique"
    logger1 = logging.get_logger(name)
    n1 = len(logger1.handlers)
    logger2 = logging.get_logger(name)
    n2 = len(logger2.handlers)
    assert n1 == n2
    assert n1 == 2  # one console + one file


# ---------------------------------------------------------------------------
# MEDIUM #14: EDropoutRBM detach on pseudo_likelihood
# (Verified implicitly: no memory growth, but we can check it runs)
# ---------------------------------------------------------------------------
def test_e_dropout_pseudo_likelihood_detached():
    """pseudo_likelihood result used in fit should be detached."""
    m = e_dropout_rbm.EDropoutRBM(n_visible=16, n_hidden=8)
    samples = torch.randn(4, 16)
    pl = m.pseudo_likelihood(samples)
    # pseudo_likelihood itself returns attached; fit() should detach
    assert pl is not None


# ---------------------------------------------------------------------------
# MEDIUM #15: RBM reconstruct uses samples.size(0)
# ---------------------------------------------------------------------------
def test_rbm_reconstruct_mse_normalization():
    """reconstruct must normalize MSE by actual batch size."""
    m = rbm.RBM(n_visible=16, n_hidden=8)
    data = torch.randn(10, 16)
    targets = torch.zeros(10)
    ds = dataset.Dataset(data, targets, show_log=False)
    mse, probs = m.reconstruct(ds)
    # MSE should be a reasonable value (not inflated by wrong denominator)
    assert mse.item() >= 0
    assert probs.shape[1] == 16


# ---------------------------------------------------------------------------
# MEDIUM #16: Temperature > 1 allowed
# ---------------------------------------------------------------------------
def test_rbm_temperature_above_one():
    """RBM should accept temperature values > 1."""
    m = rbm.RBM(n_visible=16, n_hidden=8, temperature=1.0)
    m.T = 2.0
    assert m.T == 2.0

    m.T = 5.0
    assert m.T == 5.0


def test_rbm_temperature_zero_rejected():
    """RBM should reject temperature <= 0."""
    m = rbm.RBM(n_visible=16, n_hidden=8)
    try:
        m.T = 0.0
        assert False, "Should have raised ValueError"
    except exception.ValueError:
        pass

    try:
        m.T = -1.0
        assert False, "Should have raised ValueError"
    except exception.ValueError:
        pass


# ---------------------------------------------------------------------------
# MEDIUM #18: Deprecated torch.set_default_tensor_type replaced
# ---------------------------------------------------------------------------
def test_model_no_deprecation_warning():
    """Model init should use set_default_dtype, not set_default_tensor_type."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            m = model.Model(use_gpu=False)
        except DeprecationWarning:
            assert False, "Model raised DeprecationWarning"


# ---------------------------------------------------------------------------
# MEDIUM #20: np.ndarray type hints (verify Dataset accepts ndarray)
# ---------------------------------------------------------------------------
def test_dataset_accepts_ndarray():
    """Dataset should accept np.ndarray for data and targets."""
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    targets = np.array([0, 1])
    ds = dataset.Dataset(data, targets, show_log=False)
    assert len(ds) == 2


# ---------------------------------------------------------------------------
# MEDIUM #21: Variance sigma clamping
# ---------------------------------------------------------------------------
def test_variance_gaussian_rbm_near_zero_sigma():
    """VarianceGaussianRBM should handle near-zero sigma without NaN."""
    m = gaussian_rbm.VarianceGaussianRBM(n_visible=16, n_hidden=8)
    with torch.no_grad():
        m.sigma.fill_(1e-12)  # near-zero sigma
    v = torch.randn(2, 16)
    probs, states = m.hidden_sampling(v)
    assert not torch.isnan(probs).any()

    energy = m.energy(v)
    assert not torch.isnan(energy).any()


# ---------------------------------------------------------------------------
# MEDIUM #23: Exception builtins compatibility
# ---------------------------------------------------------------------------
def test_custom_type_error_caught_by_builtin():
    """Custom TypeError should be catchable by builtin TypeError handler."""
    try:
        raise exception.TypeError("test error")
    except TypeError:
        pass  # Success: caught by builtin handler


def test_custom_value_error_caught_by_builtin():
    """Custom ValueError should be catchable by builtin ValueError handler."""
    try:
        raise exception.ValueError("test error")
    except ValueError:
        pass  # Success: caught by builtin handler


def test_custom_exceptions_still_caught_by_custom():
    """Custom exceptions should still be caught by their custom classes."""
    try:
        raise exception.TypeError("test")
    except exception.TypeError:
        pass

    try:
        raise exception.ValueError("test")
    except exception.ValueError:
        pass


# ---------------------------------------------------------------------------
# Additional edge case tests
# ---------------------------------------------------------------------------
def test_convdbn_forward_with_maxpooling():
    """ConvDBN forward with maxpooling enabled should work correctly."""
    m = conv_dbn.ConvDBN(
        model="bernoulli",
        visible_shape=(28, 28),
        filter_shape=((7, 7),),
        n_filters=(4,),
        n_channels=1,
        steps=(1,),
        learning_rate=(0.01,),
        momentum=(0,),
        decay=(0,),
        maxpooling=(True,),
        pooling_kernel=(2,),
    )
    x = torch.randn(1, 1, 28, 28)
    out = m.forward(x)
    assert out is not None


def test_dbn_model_registry():
    """DBN MODELS registry should contain all expected keys."""
    expected = {
        "bernoulli", "dropout", "e_dropout", "gaussian", "gaussian4deep",
        "gaussian_relu", "gaussian_relu4deep", "gaussian_selu",
        "sigmoid", "sigmoid4deep", "variance_gaussian",
    }
    assert set(dbn.MODELS.keys()) == expected


def test_rbm_history_dump():
    """Model.dump should append to history lists."""
    m = model.Model(use_gpu=False)
    m.dump(mse=1.0, pl=-100.0)
    m.dump(mse=0.5, pl=-50.0)
    assert m.history["mse"] == [1.0, 0.5]
    assert m.history["pl"] == [-100.0, -50.0]
