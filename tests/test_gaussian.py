import pytest
import torch
from torch.utils.data import TensorDataset

from learnergy.models.gaussian import (
    GaussianConvRBM,
    GaussianConvRBM4Deep,
    GaussianRBM,
    GaussianRBM4deep,
    GaussianReluRBM,
    GaussianReluRBM4deep,
    GaussianSeluRBM,
    VarianceGaussianRBM,
)


def test_gaussian_rbm_end_to_end():
    torch.manual_seed(0)
    dataset = TensorDataset(torch.rand(12, 16), torch.zeros(12))
    model = GaussianRBM(n_visible=16, n_hidden=8)

    mse, pl = model.fit(dataset, batch_size=4, epochs=1)
    reconstruction_mse, reconstruction = model.reconstruct(dataset)

    assert mse >= 0
    assert torch.isfinite(pl)
    assert reconstruction_mse >= 0
    assert reconstruction.shape == (12, 16)
    assert model(torch.rand(2, 16)).shape == (2, 8)


@pytest.mark.parametrize("model_class", [GaussianReluRBM, GaussianSeluRBM])
def test_gaussian_activation_variants(model_class):
    model = model_class(n_visible=16, n_hidden=8)
    probs, states = model.hidden_sampling(torch.rand(2, 16), scale=True)

    assert probs.shape == states.shape == (2, 8)
    assert torch.equal(probs, states)


def test_deep_model_names_remain_available():
    assert issubclass(GaussianRBM4deep, GaussianRBM)
    assert issubclass(GaussianReluRBM4deep, GaussianReluRBM)
    assert issubclass(GaussianConvRBM4Deep, GaussianConvRBM)


def test_variance_gaussian_rbm_sampling_is_finite():
    model = VarianceGaussianRBM(n_visible=16, n_hidden=8)
    with torch.no_grad():
        model.sigma.fill_(1e-12)

    samples = torch.rand(2, 16)
    hidden_probs, hidden_states = model.hidden_sampling(samples)
    visible_probs, visible_states = model.visible_sampling(hidden_states)

    assert torch.isfinite(hidden_probs).all()
    assert torch.isfinite(model.energy(samples)).all()
    assert visible_probs.shape == visible_states.shape == (2, 16)
    assert "sigma" in model.state_dict()
    assert len(model.optimizer.param_groups) == 2


def test_gaussian_conv_rbm_end_to_end():
    dataset = TensorDataset(torch.rand(8, 1, 8, 8), torch.zeros(8))
    model = GaussianConvRBM(
        visible_shape=(8, 8),
        filter_shape=(3, 3),
        n_filters=2,
        n_channels=1,
    )

    assert model.fit(dataset, batch_size=4, epochs=1) >= 0
    assert model(dataset.tensors[0][:2]).shape == (2, 2, 6, 6)


@pytest.mark.parametrize("model_class", [GaussianRBM, GaussianReluRBM4deep])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_gaussian_forward_standardizes_batches(model_class, batch_size):
    model = model_class(n_visible=4, n_hidden=2)
    samples = torch.arange(batch_size * 4, dtype=torch.float32).reshape(batch_size, 4)
    if batch_size == 1:
        standardized = torch.zeros_like(samples)
    else:
        standardized = (samples - samples.mean(0, True)) / (
            samples.std(0) + torch.finfo(samples.dtype).eps
        )
    expected, _ = model.hidden_sampling(standardized)

    torch.testing.assert_close(model(samples), expected, rtol=0, atol=0)


@pytest.mark.parametrize("model_class", [GaussianRBM, GaussianReluRBM4deep])
def test_gaussian_handles_singleton_training_batch_and_reconstruction(model_class):
    torch.manual_seed(0)
    model = model_class(n_visible=4, n_hidden=2)
    dataset = TensorDataset(torch.rand(5, 4), torch.zeros(5))

    mse, pl = model.fit(dataset, batch_size=2, epochs=1)
    reconstruction_mse, reconstruction = model.reconstruct(
        TensorDataset(dataset.tensors[0][:1], dataset.tensors[1][:1])
    )

    assert torch.isfinite(mse) and torch.isfinite(pl)
    assert torch.isfinite(reconstruction_mse)
    assert torch.isfinite(reconstruction).all()
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


@pytest.mark.parametrize("model_class", [GaussianConvRBM, GaussianConvRBM4Deep])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_gaussian_conv_forward_standardizes_batches(model_class, batch_size):
    model = model_class(
        visible_shape=(4, 4), filter_shape=(2, 2), n_filters=2, maxpooling=True
    )
    samples = torch.arange(batch_size * 16, dtype=torch.float32).reshape(
        batch_size, 1, 4, 4
    )
    if batch_size == 1:
        standardized = torch.zeros_like(samples)
    else:
        standardized = (samples - samples.mean(0, True)) / (
            samples.std(0) + torch.finfo(samples.dtype).eps
        )
    expected, _ = model.hidden_sampling(standardized)
    expected = model.maxpol2d(expected)

    torch.testing.assert_close(model(samples), expected, rtol=0, atol=0)


@pytest.mark.parametrize("model_class", [GaussianConvRBM, GaussianConvRBM4Deep])
def test_gaussian_conv_handles_singleton_training_batch(model_class):
    torch.manual_seed(0)
    model = model_class(visible_shape=(4, 4), filter_shape=(2, 2), n_filters=2)
    dataset = TensorDataset(torch.rand(5, 1, 4, 4), torch.zeros(5))

    mse = model.fit(dataset, batch_size=2, epochs=1)

    assert torch.isfinite(mse)
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


@pytest.mark.parametrize("model_class", [GaussianConvRBM, GaussianConvRBM4Deep])
def test_gaussian_conv_forward_preserves_gradient_flow(model_class):
    model = model_class(visible_shape=(4, 4), filter_shape=(2, 2), n_filters=2)
    with torch.no_grad():
        model.W.fill_(0.05)
        model.b.fill_(1)
    samples = torch.arange(32, dtype=torch.float32).reshape(2, 1, 4, 4)
    head = torch.nn.Linear(18, 1, bias=False)
    with torch.no_grad():
        head.weight.fill_(1)

    head(model(samples).flatten(1)).square().mean().backward()

    assert model.W.grad is not None
    assert torch.isfinite(model.W.grad).all()
    assert torch.count_nonzero(model.W.grad) > 0


def test_variance_energy_matches_marginalized_joint_distribution():
    model = VarianceGaussianRBM(n_visible=2, n_hidden=2).double()
    with torch.no_grad():
        model.W.copy_(torch.tensor([[0.2, -0.1], [0.4, 0.3]]))
        model.a.copy_(torch.tensor([0.3, -0.4]))
        model.b.copy_(torch.tensor([-0.2, 0.1]))
        model.sigma.copy_(torch.tensor([0.5, 2.0]))
    samples = torch.tensor([[0.0, 1.0], [-1.0, 2.0]], dtype=torch.float64)
    hidden = torch.tensor(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float64
    )
    variance = model.sigma.square() + torch.finfo(samples.dtype).eps
    quadratic = ((samples - model.a).square() / (2 * variance)).sum(1)
    joint_energy = (
        quadratic[:, None]
        - hidden @ model.b
        - (samples / variance) @ model.W @ hidden.t()
    )
    expected = -torch.logsumexp(-joint_energy, dim=1)

    torch.testing.assert_close(model.energy(samples), expected)


def test_variance_visible_sampling_returns_means_and_correctly_scaled_states():
    model = VarianceGaussianRBM(n_visible=2, n_hidden=1)
    with torch.no_grad():
        model.W.copy_(torch.tensor([[0.1], [0.2]]))
        model.a.copy_(torch.tensor([0.3, -0.4]))
        model.sigma.copy_(torch.tensor([0.5, 2.0]))
    hidden = torch.ones(20000, 1)

    torch.manual_seed(0)
    means, states = model.visible_sampling(hidden)
    expected_means = hidden @ model.W.t() + model.a

    torch.testing.assert_close(means, expected_means)
    torch.testing.assert_close(states.mean(0), expected_means[0], rtol=0, atol=0.04)
    torch.testing.assert_close(states.std(0), model.sigma, rtol=0.03, atol=0)


def test_variance_gibbs_sampling_uses_random_visible_states():
    model = VarianceGaussianRBM(n_visible=2, n_hidden=1)
    with torch.no_grad():
        model.W.zero_()
        model.a.zero_()

    torch.manual_seed(0)
    visible_states = model.gibbs_sampling(torch.zeros(256, 2))[-1]

    assert visible_states.std() > 0.5


def test_variance_gaussian_rbm_end_to_end():
    torch.manual_seed(0)
    model = VarianceGaussianRBM(n_visible=4, n_hidden=2, learning_rate=0.001)
    dataset = TensorDataset(torch.rand(5, 4), torch.zeros(5))
    initial_sigma = model.sigma.detach().clone()

    mse, pl = model.fit(dataset, batch_size=2, epochs=2)
    reconstruction_mse, reconstruction = model.reconstruct(dataset)

    assert torch.isfinite(mse) and torch.isfinite(pl)
    assert torch.isfinite(reconstruction_mse)
    assert reconstruction.shape == (5, 4)
    assert torch.isfinite(reconstruction).all()
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    assert not torch.equal(model.sigma, initial_sigma)
