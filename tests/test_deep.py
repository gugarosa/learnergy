from inspect import signature

import pytest
import torch
from torch.utils.data import TensorDataset

import learnergy.utils.exception as e
from learnergy.models.deep import DBN, ConvDBN, ResidualDBN
from learnergy.models.deep.dbn import MODELS
from learnergy.models.extra import SigmoidRBM


def test_dbn_builds_trains_and_reconstructs():
    dataset = TensorDataset(torch.rand(12, 16), torch.zeros(12))
    model = DBN(
        model=("bernoulli", "sigmoid"),
        n_visible=16,
        n_hidden=(8, 4),
        steps=(1, 1),
        learning_rate=(0.1, 0.1),
        momentum=(0, 0),
        decay=(0, 0),
        temperature=(1, 1),
    )

    mse, pl = model.fit(dataset, batch_size=4, epochs=(1, 1))
    reconstruction_mse, reconstruction = model.reconstruct(dataset)

    assert len(mse) == len(pl) == 2
    assert reconstruction_mse >= 0
    assert reconstruction.shape == (12, 16)
    assert model(torch.rand(2, 16)).shape == (2, 4)


def test_dbn_fills_missing_layers_with_sigmoid():
    model = DBN(
        model="bernoulli",
        n_visible=16,
        n_hidden=(8, 4),
        steps=(1, 1),
        learning_rate=(0.1, 0.1),
        momentum=(0, 0),
        decay=(0, 0),
        temperature=(1, 1),
    )

    assert isinstance(model.models[1], SigmoidRBM)
    assert set(MODELS) == {
        "bernoulli",
        "dropout",
        "e_dropout",
        "gaussian",
        "gaussian4deep",
        "gaussian_relu",
        "gaussian_relu4deep",
        "gaussian_selu",
        "sigmoid",
        "sigmoid4deep",
        "variance_gaussian",
    }


def test_dbn_validates_configuration():
    with pytest.raises(ValueError):
        DBN(model="unknown")
    with pytest.raises(e.SizeError):
        DBN(n_hidden=(8, 4), steps=(1,))


def test_dbn_forward_preserves_gradient_flow():
    model = DBN(
        model=("bernoulli", "gaussian"),
        n_visible=16,
        n_hidden=(8, 4),
        steps=(1, 1),
        learning_rate=(0.1, 0.1),
        momentum=(0, 0),
        decay=(0, 0),
        temperature=(1, 1),
    )

    model(torch.rand(2, 16)).sum().backward()
    assert model.models[0].W.grad is not None


def test_residual_dbn_end_to_end():
    dataset = TensorDataset(torch.rand(12, 16), torch.zeros(12))
    model = ResidualDBN(
        n_visible=16,
        n_hidden=(8, 4),
        steps=(1, 1),
        learning_rate=(0.1, 0.1),
        momentum=(0, 0),
        decay=(0, 0),
        temperature=(1, 1),
    )

    mse, pl = model.fit(dataset, batch_size=4, epochs=(1, 1))
    output = model(torch.rand(2, 16))

    assert len(mse) == len(pl) == 2
    assert output.shape == (2, 4)
    assert torch.isfinite(output).all()
    assert output.max() <= 1


def test_residual_dbn_validates_weights():
    with pytest.raises(ValueError):
        ResidualDBN(zetta1=-1)
    with pytest.raises(ValueError):
        ResidualDBN(zetta2=-1)


@pytest.mark.parametrize("n_samples", [8, 5])
def test_conv_dbn_builds_trains_and_reconstructs(n_samples):
    dataset = TensorDataset(torch.rand(n_samples, 1, 8, 8), torch.zeros(n_samples))
    model = ConvDBN(
        visible_shape=(8, 8),
        filter_shape=((3, 3), (3, 3)),
        n_filters=(2, 3),
        steps=(1, 1),
        learning_rate=(0.1, 0.1),
        momentum=(0, 0),
        decay=(0, 0),
    )

    mse = model.fit(dataset, batch_size=4, epochs=(1, 1))
    reconstruction_mse, reconstruction = model.reconstruct(dataset)

    assert len(mse) == 2
    assert reconstruction_mse >= 0
    assert reconstruction.shape == (n_samples, 1, 8, 8)
    assert model(dataset.tensors[0][:2]).shape == (2, 3, 4, 4)


def test_conv_dbn_pooling_configuration():
    model = ConvDBN(
        visible_shape=(8, 8),
        filter_shape=((3, 3), (2, 2)),
        n_filters=(2, 3),
        steps=(1, 1),
        learning_rate=(0.1, 0.1),
        momentum=(0, 0),
        decay=(0, 0),
        maxpooling=(True,),
        pooling_kernel=(2,),
    )

    assert model.maxpooling == (True, False)
    assert len(model.maxpol2d) == 2
    samples = torch.rand(2, 1, 8, 8)
    output = model(samples)
    assert output.shape == (2, 3, 3, 3)
    expected, _ = model.models[0].hidden_sampling(samples)
    expected = model.models[0].maxpol2d(expected)
    expected, _ = model.models[1].hidden_sampling(expected)
    assert torch.equal(output, expected)


def test_conv_dbn_accepts_legacy_defaults():
    model = ConvDBN(maxpooling=(False, False), pooling_kernel=(2, 2))
    assert model.maxpooling == (False,)
    assert model.pooling_kernel == (2,)
    assert signature(ConvDBN.fit).parameters["epochs"].default == (10, 10)


def test_gaussian_dbn_training_handles_singleton_batches():
    model = DBN(
        model=("gaussian", "gaussian"),
        n_visible=4,
        n_hidden=(3, 2),
        steps=(1, 1),
        learning_rate=(0.01, 0.01),
        momentum=(0, 0),
        decay=(0, 0),
        temperature=(1, 1),
    )
    dataset = TensorDataset(torch.rand(5, 4), torch.zeros(5))

    mse, pl = model.fit(dataset, batch_size=2, epochs=(1, 1))

    assert all(torch.isfinite(value) for value in mse + pl)
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


def test_conv_dbn_forward_preserves_gradient_flow():
    model = ConvDBN(
        visible_shape=(8, 8),
        filter_shape=((3, 3), (2, 2)),
        n_filters=(2, 3),
        steps=(1, 1),
        learning_rate=(0.1, 0.1),
        momentum=(0, 0),
        decay=(0, 0),
        maxpooling=(True, False),
    )
    with torch.no_grad():
        for layer in model.models:
            layer.W.fill_(0.05)
            layer.b.fill_(0.1)

    model(torch.ones(2, 1, 8, 8)).sum().backward()

    for layer in model.models:
        assert layer.W.grad is not None
        assert torch.isfinite(layer.W.grad).all()
        assert torch.count_nonzero(layer.W.grad) > 0
