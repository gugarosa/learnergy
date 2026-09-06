# Copyright (c) 2020-2026 Mateus Roder and Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import math

import pytest
import torch
from torch.utils.data import TensorDataset

from learnergy.models.bernoulli import (
    RBM,
    ConvRBM,
    DiscriminativeRBM,
    DropConnectRBM,
    DropoutRBM,
    EDropoutRBM,
    HybridDiscriminativeRBM,
)


def vector_dataset(samples: int = 12, features: int = 16, classes: int = 3) -> TensorDataset:
    return TensorDataset(
        torch.rand(samples, features),
        torch.randint(classes, (samples,)),
    )


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
def test_rbm_validates_constructor(kwargs):
    with pytest.raises(ValueError):
        RBM(**kwargs)


def test_rbm_end_to_end():
    torch.manual_seed(0)
    model = RBM(n_visible=16, n_hidden=8)
    samples = torch.rand(4, 16)

    hidden_probs, hidden_states = model.hidden_sampling(samples)
    visible_probs, visible_states = model.visible_sampling(hidden_states)
    gibbs = model.gibbs_sampling(samples)

    assert hidden_probs.shape == hidden_states.shape == (4, 8)
    assert visible_probs.shape == visible_states.shape == (4, 16)
    assert len(gibbs) == 5
    assert model.energy(samples).shape == (4,)
    assert torch.isfinite(model.pseudo_likelihood(samples))
    assert {"W", "a", "b"} <= model.state_dict().keys()

    mse, pl = model.fit(vector_dataset(), batch_size=4, epochs=1)
    reconstruction_mse, reconstruction = model.reconstruct(vector_dataset())

    assert mse >= 0
    assert torch.isfinite(pl)
    assert reconstruction_mse >= 0
    assert reconstruction.shape == (12, 16)
    assert model(samples).shape == (4, 8)


def test_conv_rbm_end_to_end():
    torch.manual_seed(0)
    dataset = TensorDataset(torch.rand(8, 1, 8, 8), torch.zeros(8))
    model = ConvRBM(
        visible_shape=(8, 8),
        filter_shape=(3, 3),
        n_filters=2,
        n_channels=1,
    )
    samples = dataset.tensors[0][:2]

    hidden_probs, hidden_states = model.hidden_sampling(samples)
    visible_probs, visible_states = model.visible_sampling(hidden_states)

    assert hidden_probs.shape == hidden_states.shape == (2, 2, 6, 6)
    assert visible_probs.shape == visible_states.shape == (2, 1, 8, 8)
    assert model.energy(samples).shape == (2,)
    assert model.fit(dataset, batch_size=4, epochs=1) >= 0
    assert model.reconstruct(dataset)[1].shape == (8, 1, 8, 8)


def test_conv_rbm_validates_shape_and_pooling():
    with pytest.raises(ValueError):
        ConvRBM(visible_shape=(8, 8), filter_shape=(8, 3))
    with pytest.raises(ValueError):
        ConvRBM(maxpooling=1)


@pytest.mark.parametrize("model_class", [DropoutRBM, DropConnectRBM])
def test_dropout_variants(model_class):
    model = model_class(n_visible=16, n_hidden=8, dropout=0.25)
    probs, states = model.hidden_sampling(torch.rand(4, 16))

    assert probs.shape == states.shape == (4, 8)
    assert model.p == 0.25


def test_dropout_validates_probability():
    with pytest.raises(ValueError):
        DropoutRBM(dropout=1.1)


def test_energy_dropout_handles_empty_and_zero_masks():
    model = EDropoutRBM(n_visible=16, n_hidden=8)
    samples = torch.rand(2, 16)

    probs, states = model.hidden_sampling(samples)
    assert probs.shape == states.shape == (2, 8)

    model.energy_dropout(
        torch.tensor(0.0),
        torch.zeros(2, 8),
        torch.ones(2, 8),
    )
    assert torch.isfinite(model.M).all()
    assert model.total_energy(states, samples).ndim == 0


def test_discriminative_variants():
    dataset = vector_dataset()
    model = DiscriminativeRBM(n_visible=16, n_hidden=8, n_classes=3)

    loss, accuracy = model.fit(dataset, batch_size=4, epochs=1)
    predicted_accuracy, probabilities, predictions = model.predict(dataset)

    assert loss >= 0
    assert 0 <= accuracy <= 1
    assert 0 <= predicted_accuracy <= 1
    assert probabilities.shape == (12, 3)
    assert predictions.shape == (12,)
    assert len(model.optimizer.param_groups) == 3

    hybrid = HybridDiscriminativeRBM(
        n_visible=16,
        n_hidden=8,
        n_classes=3,
        alpha=0.01,
    )
    hybrid_loss, hybrid_accuracy = hybrid.fit(dataset, batch_size=4, epochs=1)
    assert hybrid_loss >= 0
    assert 0 <= hybrid_accuracy <= 1


def test_discriminative_validates_parameters():
    with pytest.raises(ValueError):
        DiscriminativeRBM(n_classes=0)
    with pytest.raises(ValueError):
        HybridDiscriminativeRBM(alpha=-0.1)


def test_validated_attributes_remain_mutable():
    model = RBM()
    model.n_visible = 1
    assert model.n_visible == 1

    with pytest.raises(ValueError):
        model.n_visible = 0


def test_model_initialization_restores_float32_default():
    torch.set_default_dtype(torch.float64)
    try:
        model = RBM(n_visible=4, n_hidden=2)
        assert model.W.dtype == torch.float32
    finally:
        torch.set_default_dtype(torch.float32)


@pytest.mark.parametrize("bias", [-100.0, 0.0, 100.0])
def test_rbm_pseudo_likelihood_uses_stable_log_probabilities(bias):
    model = RBM(n_visible=1, n_hidden=1)
    with torch.no_grad():
        model.W.zero_()
        model.a.fill_(bias)

    log_probability = model.pseudo_likelihood(torch.zeros(1, 1))

    assert log_probability.item() == pytest.approx(-math.log1p(math.exp(bias)))
    assert log_probability <= 0
    log_probability.backward()
    assert model.a.grad.item() == pytest.approx(-1 / (1 + math.exp(-bias)))


@pytest.mark.parametrize("offset", [-1000.0, 0.0, 1000.0])
def test_hybrid_class_sampling_handles_large_logits(offset):
    model = HybridDiscriminativeRBM(n_visible=2, n_hidden=1, n_classes=3)
    with torch.no_grad():
        model.U.zero_()
        model.c.copy_(torch.tensor([offset, offset + 1, offset - 1]))

    probabilities, states = model.class_sampling(torch.zeros(2, 1))
    weights = torch.tensor([1.0, math.e, 1 / math.e])
    expected = (weights / weights.sum()).expand(2, -1)

    torch.testing.assert_close(probabilities, expected)
    torch.testing.assert_close(states, torch.tensor([[0.0, 1.0, 0.0]]).repeat(2, 1))


@pytest.mark.parametrize("model_class", [DropoutRBM, DropConnectRBM])
@pytest.mark.parametrize("shape, error", [((0, 4), ValueError), ((1, 3), RuntimeError)])
def test_dropout_restores_rate_after_reconstruction_failure(model_class, shape, error):
    model = model_class(n_visible=4, n_hidden=2, dropout=0.5)
    dataset = TensorDataset(torch.zeros(shape), torch.zeros(shape[0]))

    with pytest.raises(error):
        model.reconstruct(dataset)

    assert model.p == 0.5


@pytest.mark.parametrize("model_class", [DropoutRBM, DropConnectRBM])
def test_dropout_reconstruction_matches_disabled_dropout(model_class):
    model = model_class(n_visible=4, n_hidden=2, dropout=0.5)
    reference = model_class(n_visible=4, n_hidden=2, dropout=0)
    reference.load_state_dict(model.state_dict())
    dataset = TensorDataset(torch.rand(3, 4), torch.zeros(3))

    torch.manual_seed(12)
    expected_mse, expected_reconstruction = reference.reconstruct(dataset)
    torch.manual_seed(12)
    mse, reconstruction = model.reconstruct(dataset)

    torch.testing.assert_close(mse, expected_mse, rtol=0, atol=0)
    torch.testing.assert_close(reconstruction, expected_reconstruction, rtol=0, atol=0)
    assert model.p == 0.5
