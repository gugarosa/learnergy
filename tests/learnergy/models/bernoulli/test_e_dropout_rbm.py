import torch
import torchvision

from learnergy.models.bernoulli import e_dropout_rbm


def test_e_dropout_rbm_M():
    new_e_dropout_rbm = e_dropout_rbm.EDropoutRBM()

    assert isinstance(new_e_dropout_rbm.M, torch.Tensor)


def test_e_dropout_rbm_M_setter():
    new_e_dropout_rbm = e_dropout_rbm.EDropoutRBM()

    new_e_dropout_rbm.M = torch.Tensor()

    assert isinstance(new_e_dropout_rbm.M, torch.Tensor)


def test_e_dropout_rbm_hidden_sampling():
    new_e_dropout_rbm = e_dropout_rbm.EDropoutRBM()

    new_e_dropout_rbm.M = torch.ones((1, 128))

    v = torch.ones(1, 128)

    probs, states = new_e_dropout_rbm.hidden_sampling(v)

    assert probs.size(1) == 128
    assert states.size(1) == 128


def test_e_dropout_rbm_total_energy():
    new_e_dropout_rbm = e_dropout_rbm.EDropoutRBM()

    h = torch.ones(1, 128)
    v = torch.ones(1, 128)

    energy = new_e_dropout_rbm.total_energy(h, v)

    assert energy != 0


def test_e_dropout_rbm_energy_dropout():
    new_e_dropout_rbm = e_dropout_rbm.EDropoutRBM()

    new_e_dropout_rbm.M = torch.ones((1, 128))

    e = torch.ones(1, 128)
    p_prob = torch.ones(1, 128)
    n_prob = torch.ones(1, 128)

    new_e_dropout_rbm.energy_dropout(e, p_prob, n_prob)

    pass


def test_e_dropout_rbm_fit():
    train = torchvision.datasets.KMNIST(
        root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())

    new_e_dropout_rbm = e_dropout_rbm.EDropoutRBM(n_visible=784, n_hidden=128, steps=1,
                                                  learning_rate=0.1, momentum=0, decay=0, temperature=1, use_gpu=False)

    e, pl = new_e_dropout_rbm.fit(train, batch_size=128, epochs=1)

    assert e >= 0
    assert pl <= 0


def test_e_dropout_rbm_reconstruct():
    test = torchvision.datasets.KMNIST(
        root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())

    new_e_dropout_rbm = e_dropout_rbm.EDropoutRBM(n_visible=784, n_hidden=128, steps=1,
                                                  learning_rate=0.1, momentum=0, decay=0, temperature=1, use_gpu=False)

    e, v = new_e_dropout_rbm.reconstruct(test)

    assert e >= 0
    assert v.size(1) == 784
