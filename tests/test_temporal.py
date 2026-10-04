# Copyright (c) 2020-2026 Mateus Roder and Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import math

import pytest
import torch
from torch.utils.data import TensorDataset

import learnergy.utils.exception as e
from learnergy.models.temporal import RTDBN, RTRBM, RTGaussianRBM, RTVarianceGaussianRBM
from learnergy.models.temporal.rtdbn import RT_MODELS


@pytest.fixture(params=[RTRBM, RTGaussianRBM, RTVarianceGaussianRBM])
def model_class(request):
    return request.param


@pytest.fixture
def model(model_class):
    instance = model_class(n_visible=2, n_hidden=2, learning_rate=0.01)
    if isinstance(instance, RTGaussianRBM):
        instance.normalize = False
        instance.input_normalize = False

    return instance


def _stack(names=("variance_gaussian", "bernoulli"), **kwargs):
    configuration = {
        "model": names,
        "n_visible": 4,
        "n_hidden": (3, 2),
        "steps": (1, 1),
        "learning_rate": (0.001, 0.001),
        "momentum": (0.0, 0.0),
        "decay": (0.0, 0.0),
        "temperature": (1.0, 1.0),
        "normalize": False,
        "input_normalize": False,
    }
    configuration.update(kwargs)

    return RTDBN(**configuration)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_visible": 0},
        {"n_hidden": 0},
        {"steps": 0},
        {"learning_rate": -0.1},
        {"momentum": -0.1},
        {"decay": -0.1},
        {"temperature": 0},
    ],
)
def test_temporal_models_validate_constructor(model_class, kwargs):
    with pytest.raises(ValueError):
        model_class(**kwargs)


@pytest.mark.parametrize("name", ["n_visible", "n_hidden", "steps"])
@pytest.mark.parametrize("value", [1.5, True, "2"])
def test_temporal_models_reject_noninteger_dimensions(model_class, name, value):
    with pytest.raises(TypeError):
        model_class(**{name: value})


def test_temporal_defaults_and_parameter_registration(model_class):
    instance = model_class()
    assert instance.n_visible == instance.n_hidden == 128
    assert instance.W_prime.shape == (128, 128)
    assert torch.equal(instance.h0, torch.zeros(128))
    expected_keys = {"W", "a", "b", "W_prime", "h0"}
    if isinstance(instance, RTVarianceGaussianRBM):
        expected_keys.add("sigma")
        assert torch.equal(instance.sigma, torch.ones(128))
        assert instance.lr == 0.001
    else:
        assert instance.lr == 0.1
    assert set(instance.state_dict()) == expected_keys

    registered = {id(parameter) for parameter in instance.parameters()}
    optimized = [id(parameter) for group in instance.optimizer.param_groups for parameter in group["params"]]
    assert len(optimized) == len(registered)
    assert set(optimized) == registered


def test_temporal_models_train_reconstruct_and_generate(model):
    torch.manual_seed(74)
    dataset = TensorDataset(torch.rand(5, 3, 2), torch.arange(5))
    initial_weights = model.W.detach().clone()

    mse = model.fit(dataset, batch_size=2, epochs=2)
    reconstruction_mse, reconstruction = model.reconstruct(dataset)
    generated = model.sample(n_samples=3, n_steps=4, gibbs_steps=5)

    assert mse.ndim == reconstruction_mse.ndim == 0
    assert torch.isfinite(mse) and torch.isfinite(reconstruction_mse)
    assert not mse.requires_grad and not reconstruction_mse.requires_grad
    assert len(model.history["mse"]) == len(model.history["time"]) == 2
    assert all(isinstance(value, float) for value in model.history["mse"])
    assert reconstruction.shape == (5, 3, 2)
    assert generated.shape == (3, 4, 2)
    assert torch.isfinite(reconstruction).all() and torch.isfinite(generated).all()
    assert not generated.requires_grad
    assert not torch.equal(model.W, initial_weights)
    assert model(dataset.tensors[0]).shape == (5, 3, 2)


def test_temporal_forward_matches_mean_state_recurrence(model):
    model.double()
    with torch.no_grad():
        model.W.copy_(torch.tensor([[0.2, -0.1], [0.4, 0.3]]))
        model.W_prime.copy_(torch.tensor([[0.5, -0.2], [-0.3, 0.4]]))
        model.b.copy_(torch.tensor([-0.2, 0.1]))
        model.h0.copy_(torch.tensor([-0.3, 0.2]))
        if isinstance(model, RTVarianceGaussianRBM):
            model.sigma.copy_(torch.tensor([0.5, 2.0]))
    samples = torch.tensor([[[0.1, 0.5], [0.8, 0.2], [-0.4, 0.6]]], dtype=torch.float64, requires_grad=True)
    previous = model.h0.unsqueeze(0)
    expected = []
    for visible in samples.unbind(1):
        if isinstance(model, RTVarianceGaussianRBM):
            visible = visible / (model.sigma.square() + torch.finfo(visible.dtype).eps)
        previous = torch.sigmoid(visible @ model.W + previous @ model.W_prime.t() + model.b)
        expected.append(previous)

    actual = model(samples)

    torch.testing.assert_close(actual, torch.stack(expected, dim=1))
    torch.testing.assert_close(model(samples), actual)
    actual.sum().backward()
    assert samples.grad is not None
    assert model.W_prime.grad.abs().sum() > 0
    assert model.h0.grad.abs().sum() > 0


def test_temporal_energy_matches_marginalized_joint_distribution(model):
    model.double()
    with torch.no_grad():
        model.W.copy_(torch.tensor([[0.2, -0.1], [0.4, 0.3]]))
        model.W_prime.copy_(torch.tensor([[0.3, -0.1], [0.2, 0.4]]))
        model.a.copy_(torch.tensor([0.3, -0.4]))
        model.b.copy_(torch.tensor([-0.2, 0.1]))
        if isinstance(model, RTVarianceGaussianRBM):
            model.sigma.copy_(torch.tensor([0.5, 2.0]))
    samples = torch.tensor([[0.0, 1.0], [-1.0, 2.0]], dtype=torch.float64)
    previous = torch.tensor([[0.2, 0.6], [-0.1, 0.4]], dtype=torch.float64)
    hidden = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
    variance = model.sigma.square() + torch.finfo(samples.dtype).eps if hasattr(model, "sigma") else 1
    if type(model) is RTRBM:
        visible_energy = -samples @ model.a
    else:
        visible_energy = ((samples - model.a).square() / (2 * variance)).sum(1)
    bias = previous @ model.W_prime.t() + model.b
    joint_energy = visible_energy[:, None] - bias @ hidden.t() - (samples / variance) @ model.W @ hidden.t()
    expected = -torch.logsumexp(-joint_energy, dim=1)

    torch.testing.assert_close(model.energy(samples, previous), expected)


def test_temporal_hidden_sampling_matches_energy_context_and_temperature(model):
    with torch.no_grad():
        model.W.zero_()
        model.W_prime.fill_(1)
        model.b.fill_(0.2)
        model.h0.fill_(-0.3)
    model.T = 0.5
    visible = torch.zeros(2, 2)
    previous = -torch.ones(2, 2)
    probabilities, states = model.hidden_sampling(visible, previous, scale=True)

    torch.testing.assert_close(probabilities, torch.sigmoid(torch.full((2, 2), -1.8 / 0.5)))
    assert torch.all((states == 0) | (states == 1))
    torch.testing.assert_close(model.pre_activation(visible), torch.full((2, 2), -0.4))
    torch.testing.assert_close(model.energy(visible), model.energy(visible, model.h0.expand(2, -1)))


def test_temporal_training_differentiates_through_time_and_detaches_particles(model, monkeypatch):
    model.double()
    with torch.no_grad():
        model.W.fill_(0.3)
        model.W_prime.fill_(0.4)
        model.a.fill_(0.1)
        model.b.fill_(0.2)
        model.h0.fill_(0.25)
    sequence = torch.tensor([[[0.2, -0.4], [0.3, 0.1], [-0.2, 0.5]]], dtype=torch.float64, requires_grad=True)
    negative = torch.full((1, 2), 0.125, dtype=torch.float64, requires_grad=True)
    hidden = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
    variance = model.sigma.square() + torch.finfo(sequence.dtype).eps if hasattr(model, "sigma") else 1
    previous = model.h0.unsqueeze(0)
    expected_cost = sequence.new_zeros(())
    for visible in sequence.unbind(1):
        energies = []
        for observation in (visible, negative.detach()):
            if type(model) is RTRBM:
                visible_energy = -observation @ model.a
            else:
                visible_energy = ((observation - model.a).square() / (2 * variance)).sum(1)
            activation = (observation / variance) @ model.W + previous @ model.W_prime.t() + model.b
            energies.append(-torch.logsumexp(activation @ hidden.t() - visible_energy[:, None], dim=1).mean())
        expected_cost = expected_cost + energies[0] - energies[1]
        previous = torch.sigmoid((visible / variance) @ model.W + previous @ model.W_prime.t() + model.b)
    parameters = tuple(model.parameters())
    gradients = torch.autograd.grad(expected_cost, parameters)
    if type(model) is not RTRBM:
        norm = torch.stack([gradient.norm() for gradient in gradients]).norm()
        coefficient = torch.clamp(1 / (norm + 1e-6), max=1)
        gradients = tuple(gradient * coefficient for gradient in gradients)

    def gibbs_sampling(visible, previous):
        return None, None, None, None, negative

    monkeypatch.setattr(model, "gibbs_sampling", gibbs_sampling)
    mse = model.fit_subseries(sequence)

    for parameter, expected in zip(parameters, gradients):
        torch.testing.assert_close(parameter.grad, expected)
    torch.testing.assert_close(mse, (sequence.detach() - negative.detach().unsqueeze(1)).square().sum())
    assert negative.grad is None
    assert sequence.grad is not None


def test_temporal_cd_step_updates_recurrent_parameters(model):
    torch.manual_seed(74)
    previous_weights = model.W_prime.detach().clone()
    previous = torch.full((3, 2), 0.5)

    mse = model.cd_step(torch.rand(3, 2), previous)

    assert torch.isfinite(mse) and not mse.requires_grad
    assert not torch.equal(model.W_prime, previous_weights)
    assert model.history == {}


def test_temporal_gibbs_steps_use_visible_states_and_fixed_context(model, monkeypatch):
    model.steps = 3
    visible_calls = []
    hidden_calls = []
    previous = torch.full((2, 2), 0.4)

    def hidden_sampling(visible, h_prev, scale=False):
        hidden_calls.append((visible, h_prev, scale))
        return torch.full_like(visible, 0.5), torch.zeros_like(visible)

    def visible_sampling(hidden, scale=False):
        visible_calls.append(scale)
        return torch.zeros_like(hidden), torch.full_like(hidden, len(visible_calls))

    monkeypatch.setattr(model, "hidden_sampling", hidden_sampling)
    monkeypatch.setattr(model, "visible_sampling", visible_sampling)

    *_, states = model.gibbs_sampling(torch.zeros(2, 2), previous)

    assert visible_calls == [True, True, True]
    assert len(hidden_calls) == 4
    for i, (visible, context, scale) in enumerate(hidden_calls):
        torch.testing.assert_close(visible, torch.full_like(visible, i))
        assert context is previous
        assert scale is (i > 0)
    torch.testing.assert_close(states, torch.full_like(states, 3))


def test_temporal_generation_uses_requested_steps_and_mean_recurrent_context(model, monkeypatch):
    visible_calls = []
    contexts = []

    def sample_visible(hidden):
        visible_calls.append(hidden)
        return torch.full_like(hidden, len(visible_calls) / 10)

    def hidden_sampling(visible, h_prev):
        contexts.append(h_prev.clone())
        return torch.sigmoid(visible), torch.zeros_like(visible)

    monkeypatch.setattr(model, "_sample_visible", sample_visible)
    monkeypatch.setattr(model, "hidden_sampling", hidden_sampling)

    generated = model.sample(n_samples=2, n_steps=2, gibbs_steps=2)

    assert len(visible_calls) == 4
    assert len(contexts) == 6
    torch.testing.assert_close(generated[:, 0], torch.full((2, 2), 0.2))
    torch.testing.assert_close(generated[:, 1], torch.full((2, 2), 0.4))
    for context in contexts[:3]:
        torch.testing.assert_close(context, model.h0.expand(2, -1))
    for context in contexts[3:]:
        torch.testing.assert_close(context, torch.sigmoid(torch.full((2, 2), 0.2)))


def test_temporal_pseudo_likelihood_holds_context_fixed():
    model = RTRBM(n_visible=1, n_hidden=1)
    with torch.no_grad():
        model.W.fill_(0.7)
        model.a.fill_(0.3)
        model.b.fill_(-0.2)
        model.W_prime.fill_(0.9)
    hidden_bias = -0.2 + 0.9 * 0.6
    log_odds = 0.3 + math.log1p(math.exp(0.7 + hidden_bias)) - math.log1p(math.exp(hidden_bias))

    likelihood = model.pseudo_likelihood(torch.zeros(1, 1), torch.full((1, 1), 0.6))

    assert likelihood.item() == pytest.approx(-math.log1p(math.exp(log_odds)))
    assert torch.isfinite(model.pseudo_likelihood(torch.zeros(1, 1)))
    likelihood.backward()
    assert torch.isfinite(model.W_prime.grad).all()


@pytest.mark.parametrize("operation", ["forward", "fit_subseries"])
@pytest.mark.parametrize("shape", [(2, 2), (0, 3, 2), (1, 0, 2), (1, 3, 3)])
def test_temporal_models_reject_invalid_sequence_shapes(model, operation, shape):
    with pytest.raises(e.SizeError):
        getattr(model, operation)(torch.zeros(shape))


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_temporal_models_reject_nonfinite_sequences(model, value):
    with pytest.raises(ValueError):
        model(torch.full((1, 2, 2), value))


def test_temporal_models_reject_integer_sequences(model):
    with pytest.raises(TypeError):
        model(torch.ones(1, 2, 2, dtype=torch.int64))


def test_temporal_models_reject_invalid_context_shapes(model):
    with pytest.raises(e.SizeError):
        model.hidden_sampling(torch.zeros(2, 2), torch.zeros(1, 2))


@pytest.mark.parametrize("name", ["n_samples", "n_steps", "gibbs_steps"])
@pytest.mark.parametrize("value, error", [(0, ValueError), (-1, ValueError), (1.5, TypeError), (True, TypeError)])
def test_temporal_sampling_validates_counts(model, name, value, error):
    with pytest.raises(error):
        model.sample(**{name: value})


def test_temporal_training_and_reconstruction_reject_empty_data(model):
    empty = TensorDataset(torch.empty(0, 3, 2), torch.empty(0))
    with pytest.raises(ValueError):
        model.fit(empty)
    with pytest.raises(ValueError):
        model.reconstruct(empty)
    dataset = TensorDataset(torch.zeros(1, 3, 2), torch.zeros(1))
    with pytest.raises(ValueError):
        model.fit(dataset, epochs=0)
    with pytest.raises(ValueError):
        model.fit(dataset, batch_size=0)
    assert model.history == {}


def test_temporal_reconstruction_returns_conditional_values(model):
    with torch.no_grad():
        model.W.zero_()
        model.a.copy_(torch.tensor([-2.0, 3.0]))
    dataset = TensorDataset(torch.ones(2, 3, 2), torch.zeros(2))

    _, reconstruction = model.reconstruct(dataset)

    expected = torch.sigmoid(model.a) if type(model) is RTRBM else model.a
    torch.testing.assert_close(reconstruction, expected.expand(2, 3, -1))


def test_temporal_sampling_matches_independent_visible_distribution(model):
    with torch.no_grad():
        model.W.zero_()
        model.a.copy_(torch.tensor([0.3, -0.4]))
        if isinstance(model, RTVarianceGaussianRBM):
            model.sigma.copy_(torch.tensor([0.5, 2.0]))
    torch.manual_seed(74)

    samples = model.sample(n_samples=20000, n_steps=1, gibbs_steps=2)[:, 0]

    if type(model) is RTRBM:
        assert torch.all((samples == 0) | (samples == 1))
        torch.testing.assert_close(samples.mean(0), torch.sigmoid(model.a), rtol=0, atol=0.02)
    else:
        expected_std = model.sigma if hasattr(model, "sigma") else torch.ones(2)
        torch.testing.assert_close(samples.mean(0), model.a, rtol=0, atol=0.04)
        torch.testing.assert_close(samples.std(0), expected_std, rtol=0.03, atol=0)


def test_temporal_sampling_preserves_dtype_and_module_state(model):
    model.double().eval()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    first = model.sample(n_samples=2, n_steps=3, gibbs_steps=2)
    second = model.sample(n_samples=2, n_steps=3, gibbs_steps=2)

    assert first.dtype == second.dtype == torch.float64
    assert first.device == model.W.device
    assert not first.requires_grad
    assert model.training is False
    assert model.history == {}
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)


def test_temporal_checkpoint_and_optimizer_round_trip(model, model_class):
    torch.manual_seed(74)
    model.fit_subseries(torch.rand(2, 3, 2))
    restored = model_class(n_visible=2, n_hidden=2, learning_rate=0.01)
    restored.load_state_dict(model.state_dict())
    restored.optimizer.load_state_dict(model.optimizer.state_dict())
    torch.manual_seed(12)
    expected = model.sample(n_samples=2, n_steps=3, gibbs_steps=3)
    torch.manual_seed(12)

    torch.testing.assert_close(restored.sample(n_samples=2, n_steps=3, gibbs_steps=3), expected, rtol=0, atol=0)
    assert len(restored.optimizer.param_groups) == len(model.optimizer.param_groups)


@pytest.mark.parametrize("shape", [(1, 1, 2), (1, 3, 2), (2, 3, 2)])
def test_temporal_gaussian_standardizes_combined_batch_and_time(shape):
    model = RTGaussianRBM(n_visible=2, n_hidden=2)
    samples = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape).requires_grad_()
    flat = samples.detach().reshape(-1, 2)
    if len(flat) == 1:
        standardized = torch.zeros_like(samples)
    else:
        standardized = ((flat - flat.mean(0)) / (flat.std(0) + torch.finfo(flat.dtype).eps)).reshape(shape)
    model.input_normalize = False
    expected = model(standardized)
    model.input_normalize = True

    actual = model(samples)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    assert samples.grad is None
    assert model.W.grad is not None


def test_temporal_gaussian_normalization_controls_are_independent(monkeypatch):
    model = RTGaussianRBM(n_visible=2, n_hidden=2, normalize=True, input_normalize=False)
    samples = torch.tensor([[[2.0, 4.0]]], requires_grad=True)
    model(samples).sum().backward()
    assert samples.grad is not None
    observed = []

    def fit_subseries(instance, sequence):
        observed.append(sequence)
        return sequence.new_zeros(())

    monkeypatch.setattr(RTRBM, "fit_subseries", fit_subseries)
    model.fit_subseries(samples)
    torch.testing.assert_close(observed[0], torch.zeros_like(samples))
    assert observed[0].requires_grad is False


def test_temporal_gaussian_singleton_training_and_reconstruction_are_finite():
    torch.manual_seed(74)
    model = RTGaussianRBM(n_visible=2, n_hidden=2)
    dataset = TensorDataset(torch.ones(1, 1, 2), torch.zeros(1))

    mse = model.fit(dataset, batch_size=1, epochs=1)
    reconstruction_mse, reconstruction = model.reconstruct(dataset)

    assert torch.isfinite(mse) and torch.isfinite(reconstruction_mse)
    assert torch.isfinite(reconstruction).all()
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


def test_temporal_gaussian_gibbs_returns_continuous_states():
    model = RTGaussianRBM(n_visible=2, n_hidden=2, temperature=2)
    with torch.no_grad():
        model.W.zero_()
        model.a.copy_(torch.tensor([-4.0, 6.0]))

    *_, states = model.gibbs_sampling(torch.zeros(3, 2), torch.zeros(3, 2))

    torch.testing.assert_close(states, torch.tensor([-2.0, 3.0]).expand(3, -1))


def test_temporal_variance_sampling_returns_means_and_correct_standard_deviation():
    model = RTVarianceGaussianRBM(n_visible=2, n_hidden=1)
    with torch.no_grad():
        model.W.copy_(torch.tensor([[0.1], [0.2]]))
        model.a.copy_(torch.tensor([0.3, -0.4]))
        model.sigma.copy_(torch.tensor([0.5, 2.0]))
    hidden = torch.ones(20000, 1)
    torch.manual_seed(74)

    means, states = model.visible_sampling(hidden)

    torch.testing.assert_close(means, hidden @ model.W.t() + model.a)
    torch.testing.assert_close(states.mean(0), means[0], rtol=0, atol=0.04)
    torch.testing.assert_close(states.std(0), model.sigma, rtol=0.03, atol=0)
    torch.manual_seed(74)
    scaled_means, scaled_states = model.visible_sampling(hidden, scale=True)
    torch.testing.assert_close(scaled_means, means, rtol=0, atol=0)
    torch.testing.assert_close(scaled_states, states, rtol=0, atol=0)


def test_temporal_variance_gibbs_uses_random_particles_in_both_negative_outputs():
    model = RTVarianceGaussianRBM(n_visible=2, n_hidden=2)
    with torch.no_grad():
        model.W.copy_(torch.eye(2))
    previous = torch.zeros(256, 2)
    torch.manual_seed(74)

    _, _, probabilities, _, visible_states = model.gibbs_sampling(torch.zeros(256, 2), previous)

    expected = torch.sigmoid(visible_states / (model.sigma.square() + torch.finfo(visible_states.dtype).eps))
    torch.testing.assert_close(probabilities, expected)
    assert visible_states.std() > 0.5


@pytest.mark.parametrize("sigma", [0.0, 1e-12, -0.5])
def test_temporal_variance_stays_finite_with_small_or_signed_scales(sigma):
    model = RTVarianceGaussianRBM(n_visible=2, n_hidden=2)
    with torch.no_grad():
        model.sigma.fill_(sigma)
    samples = torch.ones(2, 2)
    probabilities, states = model.hidden_sampling(samples)
    means, visible_states = model.visible_sampling(states)

    assert torch.isfinite(probabilities).all()
    assert torch.isfinite(means).all() and torch.isfinite(visible_states).all()
    assert torch.isfinite(model.energy(samples)).all()


def test_temporal_variance_training_preserves_frozen_scales():
    model = RTVarianceGaussianRBM(n_visible=2, n_hidden=2)
    with torch.no_grad():
        model.sigma.fill_(0.05)
    model.sigma.requires_grad_(False)

    model.fit_subseries(torch.rand(2, 3, 2))

    torch.testing.assert_close(model.sigma, torch.full((2,), 0.05), rtol=0, atol=0)
    assert model.sigma.requires_grad is False
    assert model.sigma.grad is None


@pytest.mark.parametrize("model_class", [RTGaussianRBM, RTVarianceGaussianRBM])
def test_temporal_gaussian_training_clips_gradients(model_class):
    torch.manual_seed(74)
    model = model_class(n_visible=2, n_hidden=2)

    model.fit_subseries(torch.randn(2, 5, 2) * 5)

    norm = torch.stack([parameter.grad.norm() for parameter in model.parameters()]).norm()
    assert norm <= 1.00001
    if isinstance(model, RTVarianceGaussianRBM):
        assert torch.all((model.sigma >= 0.1) & (model.sigma <= 10))


def test_rtdbn_defaults_and_supported_models():
    model = RTDBN()
    assert model.n_visible == 78
    assert model.n_hidden == (64,)
    assert model.n_layers == len(model.models) == 1
    assert isinstance(model.models[0], RTVarianceGaussianRBM)
    assert RT_MODELS == {"bernoulli": RTRBM, "gaussian": RTGaussianRBM, "variance_gaussian": RTVarianceGaussianRBM}
    assert set(model.state_dict()) == {
        "models.0.W",
        "models.0.a",
        "models.0.b",
        "models.0.W_prime",
        "models.0.h0",
        "models.0.sigma",
    }


@pytest.mark.parametrize("name", ["bernoulli", "gaussian", "variance_gaussian"])
def test_rtdbn_supports_each_advertised_family(name):
    torch.manual_seed(74)
    model = RTDBN(model=name, n_visible=2, n_hidden=(2,))
    dataset = TensorDataset(torch.rand(5, 3, 2), torch.arange(5))

    errors = model.fit(dataset, batch_size=2, epochs=(1,))

    assert len(errors) == 1 and isinstance(errors[0], float)
    assert len(model.models[0].history["mse"]) == 1
    assert model(dataset.tensors[0]).shape == (5, 2)
    torch.manual_seed(12)
    expected = model.models[0].sample(n_samples=2, n_steps=3, gibbs_steps=3)
    torch.manual_seed(12)
    torch.testing.assert_close(model.sample(n_samples=2, n_steps=3, gibbs_steps=3), expected, rtol=0, atol=0)


@pytest.mark.parametrize("name", ["model", "steps", "learning_rate", "momentum", "decay", "temperature"])
def test_rtdbn_validates_layerwise_option_lengths(name):
    value = ("bernoulli",) if name == "model" else (1,)
    with pytest.raises(e.SizeError):
        _stack(**{name: value})


@pytest.mark.parametrize("hidden", [(), (0,), (-1,)])
def test_rtdbn_rejects_invalid_hidden_dimensions(hidden):
    with pytest.raises(ValueError):
        RTDBN(n_hidden=hidden)


def test_rtdbn_rejects_unknown_models():
    with pytest.raises(ValueError, match="unknown model"):
        RTDBN(model="unknown")


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"epochs": (1,)}, e.SizeError),
        ({"epochs": (0, 1)}, ValueError),
        ({"epochs": (1, 1.5)}, TypeError),
        ({"warmup_epochs": (1, 1, 1)}, e.SizeError),
        ({"warmup_epochs": (-1,)}, ValueError),
        ({"warmup_epochs": (True,)}, TypeError),
        ({"batch_size": 0}, ValueError),
    ],
)
def test_rtdbn_validates_training_schedule_before_updates(kwargs, error):
    model = _stack()
    dataset = TensorDataset(torch.rand(2, 3, 4), torch.zeros(2))
    options = {"epochs": (1, 1)}
    options.update(kwargs)

    with pytest.raises(error):
        model.fit(dataset, **options)
    assert all(layer.history == {} for layer in model.models)


@pytest.mark.parametrize(
    "epochs, warmup, expected", [(1, 15, [(1, False)]), (3, 1, [(1, False), (2, True)]), (2, 0, [(2, True)])]
)
def test_rtdbn_warmup_respects_the_total_epoch_budget(monkeypatch, epochs, warmup, expected):
    model = RTDBN(n_visible=2, n_hidden=(2,))
    calls = []

    def fit(dataset, batch_size, epochs):
        calls.append((epochs, model.models[0].sigma.requires_grad))
        model.models[0].dump(mse=0.0)
        return torch.tensor(0.0)

    monkeypatch.setattr(model.models[0], "fit", fit)
    dataset = TensorDataset(torch.zeros(2, 3, 2), torch.zeros(2))

    assert model.fit(dataset, epochs=(epochs,), warmup_epochs=(warmup,)) == [0.0]
    assert calls == expected
    assert model.models[0].sigma.requires_grad is True


@pytest.mark.parametrize("requires_grad", [False, True])
def test_rtdbn_restores_scale_freezes_after_training_failure(monkeypatch, requires_grad):
    model = RTDBN(n_visible=2, n_hidden=(2,))
    model.models[0].sigma.requires_grad_(requires_grad)
    model.eval()

    def fit(dataset, batch_size, epochs):
        assert model.models[0].sigma.requires_grad is False
        raise RuntimeError("training interrupted")

    monkeypatch.setattr(model.models[0], "fit", fit)
    dataset = TensorDataset(torch.zeros(2, 3, 2), torch.zeros(2))

    with pytest.raises(RuntimeError, match="training interrupted"):
        model.fit(dataset, epochs=(1,))
    assert model.models[0].sigma.requires_grad is requires_grad
    assert model.training is False and model.models[0].training is False


def test_rtdbn_encodes_detached_features_and_preserves_targets_and_freezes(monkeypatch):
    torch.manual_seed(74)
    model = _stack()
    model.eval()
    model.models[0].b.requires_grad_(False)
    dataset = TensorDataset(torch.rand(5, 3, 4), torch.arange(5))
    batches = []
    handle = model.models[0].register_forward_hook(lambda module, args, output: batches.append(output))

    def fit(encoded, batch_size, epochs):
        with torch.no_grad():
            expected = model.models[0].forward(dataset.tensors[0])
        torch.testing.assert_close(encoded.tensors[0], expected)
        torch.testing.assert_close(encoded.tensors[1], dataset.tensors[1])
        assert encoded.tensors[0].device.type == "cpu"
        assert encoded.tensors[0].requires_grad is False
        model.models[1].dump(mse=0.0)
        return torch.tensor(0.0)

    monkeypatch.setattr(model.models[1], "fit", fit)
    try:
        errors = model.fit(dataset, batch_size=2, epochs=(1, 1), warmup_epochs=())
    finally:
        handle.remove()

    assert len(errors) == 2
    assert len(batches) == 3
    assert all(batch.requires_grad is False for batch in batches)
    assert model.models[0].b.requires_grad is False
    assert all(layer.training is False for layer in model.models)


@pytest.mark.parametrize("first", ["bernoulli", "gaussian", "variance_gaussian"])
def test_rtdbn_trains_every_layer_and_generates_sequences(first):
    torch.manual_seed(74)
    model = _stack((first, "bernoulli"))
    dataset = TensorDataset(torch.rand(5, 3, 4), torch.arange(5))
    before = [layer.W.detach().clone() for layer in model.models]

    errors = model.fit(dataset, batch_size=2, epochs=(1, 1), warmup_epochs=())
    generated = model.sample(n_samples=2, n_steps=3, gibbs_steps=5)

    assert len(errors) == 2
    assert all(math.isfinite(error) for error in errors)
    for layer, weights in zip(model.models, before):
        assert len(layer.history["mse"]) == 1
        assert not torch.equal(layer.W, weights)
    assert generated.shape == (2, 3, 4)
    assert torch.isfinite(generated).all()


@pytest.mark.parametrize("first", ["bernoulli", "gaussian", "variance_gaussian"])
def test_rtdbn_multilayer_encoding_preserves_time_until_pooling_and_gradients(first):
    torch.manual_seed(74)
    model = _stack((first, "bernoulli"))
    samples = torch.rand(2, 3, 4, requires_grad=True)
    expected = model.models[1](model.models[0](samples)).mean(1)

    actual = model.encode(samples)

    assert actual.shape == (2, 2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    assert samples.grad is not None
    assert all(layer.W.grad is not None and torch.isfinite(layer.W.grad).all() for layer in model.models)


def test_rtdbn_passes_normalization_flags_only_to_fixed_gaussian_layers():
    model = _stack(("gaussian", "variance_gaussian"), normalize=False, input_normalize=True)
    assert model.models[0].normalize is False
    assert model.models[0].input_normalize is True
    assert not hasattr(model.models[1], "normalize")


@pytest.mark.parametrize("first", ["bernoulli", "gaussian", "variance_gaussian"])
def test_rtdbn_multilayer_sampling_draws_lower_visible_conditionals(first):
    model = _stack((first, "bernoulli"))
    with torch.no_grad():
        model.models[0].W.zero_()
        model.models[0].a.fill_(0.3)
        if first == "variance_gaussian":
            model.models[0].sigma.fill_(2)
    torch.manual_seed(74)

    samples = model.sample(n_samples=10000, n_steps=2, gibbs_steps=2)

    assert samples.shape == (10000, 2, 4)
    assert not samples.requires_grad
    if first == "bernoulli":
        assert torch.all((samples == 0) | (samples == 1))
        assert samples.mean().item() == pytest.approx(torch.sigmoid(torch.tensor(0.3)).item(), abs=0.015)
    else:
        assert samples.mean().item() == pytest.approx(0.3, abs=0.04)
        assert samples.std().item() == pytest.approx(2 if first == "variance_gaussian" else 1, rel=0.03)


def test_rtrbm_generation_learns_alternating_temporal_structure():
    torch.manual_seed(3)
    data = torch.zeros(200, 6, 6)
    data[:, 0::2, :3] = 1
    data[:, 1::2, 3:] = 1
    data = torch.where(torch.rand_like(data) < 0.03, 1 - data, data)
    dataset = TensorDataset(data, torch.zeros(len(data)))
    torch.manual_seed(3)
    model = RTRBM(n_visible=6, n_hidden=12, learning_rate=0.05)

    model.fit(dataset, batch_size=32, epochs=40)
    samples = model.sample(n_samples=100, n_steps=6, gibbs_steps=50)

    assert samples[:, 0::2, :3].mean() - samples[:, 0::2, 3:].mean() > 0.2
    assert samples[:, 1::2, 3:].mean() - samples[:, 1::2, :3].mean() > 0.2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_temporal_models_train_and_sample_on_cuda(model_class):
    model = model_class(n_visible=2, n_hidden=2, use_gpu=True)
    dataset = TensorDataset(torch.rand(5, 3, 2), torch.zeros(5))

    mse = model.fit(dataset, batch_size=2, epochs=1)
    generated = model.sample(n_samples=2, n_steps=3, gibbs_steps=2)

    assert mse.device.type == generated.device.type == "cuda"
    assert torch.isfinite(mse) and torch.isfinite(generated).all()
    registered = {id(parameter) for parameter in model.parameters()}
    optimized = {id(parameter) for group in model.optimizer.param_groups for parameter in group["params"]}
    assert optimized == registered
