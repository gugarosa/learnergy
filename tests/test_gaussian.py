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
