import torch
import torchvision

from learnergy.models.bernoulli import discriminative_rbm


def test_discriminative_rbm_n_classes():
    new_discriminative_rbm = discriminative_rbm.DiscriminativeRBM()

    assert new_discriminative_rbm.n_classes == 1


def test_discriminative_rbm_n_classes_setter():
    new_discriminative_rbm = discriminative_rbm.DiscriminativeRBM()

    try:
        new_discriminative_rbm.n_classes = 'a'
    except:
        new_discriminative_rbm.n_classes = 1

    assert new_discriminative_rbm.n_classes == 1

    try:
        new_discriminative_rbm.n_classes = 0
    except:
        new_discriminative_rbm.n_classes = 1

    assert new_discriminative_rbm.n_classes == 1


def test_discriminative_rbm_U():
    new_discriminative_rbm = discriminative_rbm.DiscriminativeRBM()

    assert new_discriminative_rbm.U.size(0) == 1
    assert new_discriminative_rbm.U.size(1) == 128


def test_discriminative_rbm_U_setter():
    new_discriminative_rbm = discriminative_rbm.DiscriminativeRBM()

    try:
        new_discriminative_rbm.U = 1
    except:
        new_discriminative_rbm.U = torch.nn.Parameter(
            torch.randn(10, 128) * 0.01)

    assert new_discriminative_rbm.U.size(0) == 10
    assert new_discriminative_rbm.U.size(1) == 128


def test_discriminative_rbm_c():
    new_discriminative_rbm = discriminative_rbm.DiscriminativeRBM()

    assert new_discriminative_rbm.c.size(0) == 1


def test_discriminative_rbm_c_setter():
    new_discriminative_rbm = discriminative_rbm.DiscriminativeRBM()

    try:
        new_discriminative_rbm.c = 1
    except:
        new_discriminative_rbm.c = torch.nn.Parameter(torch.zeros(10))

    assert new_discriminative_rbm.c.size(0) == 10


def test_discriminative_rbm_loss():
    new_discriminative_rbm = discriminative_rbm.DiscriminativeRBM()

    assert type(new_discriminative_rbm.loss).__name__ == 'CrossEntropyLoss'


def test_discriminative_rbm_loss_setter():
    new_discriminative_rbm = discriminative_rbm.DiscriminativeRBM()

    try:
        new_discriminative_rbm.loss = 'LOSS'
    except:
        new_discriminative_rbm.loss = torch.nn.CrossEntropyLoss()

    assert type(new_discriminative_rbm.loss).__name__ == 'CrossEntropyLoss'


def test_discriminative_rbm_labels_sampling():
    new_discriminative_rbm = discriminative_rbm.DiscriminativeRBM()

    samples = torch.ones(1, 128)

    probs, labels = new_discriminative_rbm.labels_sampling(samples)

    assert probs.size(1) == 1
    assert labels.size(0) == 1


def test_discriminative_rbm_fit():
    train = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())

    new_discriminative_rbm = discriminative_rbm.DiscriminativeRBM(n_visible=784, n_hidden=128, n_classes=10,
                                                                  learning_rate=0.1, momentum=0, decay=0, use_gpu=False)

    loss, acc = new_discriminative_rbm.fit(train, batch_size=128, epochs=1)

    assert loss >= 0
    assert acc >= 0


def test_discriminative_rbm_predict():
    test = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())

    new_discriminative_rbm = discriminative_rbm.DiscriminativeRBM(n_visible=784, n_hidden=128, n_classes=10,
                                                                  learning_rate=0.1, momentum=0, decay=0, use_gpu=False)

    acc, probs, labels = new_discriminative_rbm.predict(test)

    assert acc >= 0
    assert probs.size(0) == 10000
    assert labels.size(0) == 10000


def test_hybrid_discriminative_rbm_alpha():
    new_hybrid_discriminative_rbm = discriminative_rbm.HybridDiscriminativeRBM()

    assert new_hybrid_discriminative_rbm.alpha == 0.01


def test_hybrid_discriminative_rbm_alpha_setter():
    new_hybrid_discriminative_rbm = discriminative_rbm.HybridDiscriminativeRBM()

    try:
        new_hybrid_discriminative_rbm.alpha = 'a'
    except:
        new_hybrid_discriminative_rbm.alpha = 0.01

    assert new_hybrid_discriminative_rbm.alpha == 0.01

    try:
        new_hybrid_discriminative_rbm.alpha = -1
    except:
        new_hybrid_discriminative_rbm.alpha = 0.01

    assert new_hybrid_discriminative_rbm.alpha == 0.01


def test_hybrid_discriminative_rbm_hidden_sampling():
    new_hybrid_discriminative_rbm = discriminative_rbm.HybridDiscriminativeRBM(
        n_classes=10)

    v = torch.ones(1, 128)
    y = torch.ones(128, 10)

    probs, states = new_hybrid_discriminative_rbm.hidden_sampling(v, y)

    assert probs.size(1) == 128
    assert states.size(1) == 128


def test_hybrid_discriminative_rbm_class_sampling():
    new_hybrid_discriminative_rbm = discriminative_rbm.HybridDiscriminativeRBM(
        n_classes=10)

    h = torch.ones(1, 128)

    probs, states = new_hybrid_discriminative_rbm.class_sampling(h)

    assert probs.size(1) == 10
    assert states.size(1) == 10


def test_hybrid_discriminative_rbm_fit():
    train = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())

    new_hybrid_discriminative_rbm = discriminative_rbm.HybridDiscriminativeRBM(n_visible=784, n_hidden=128, n_classes=10,
                                                                               learning_rate=0.1, alpha=0.01, momentum=0, decay=0, use_gpu=False)

    loss, acc = new_hybrid_discriminative_rbm.fit(
        train, batch_size=128, epochs=1)

    assert loss >= 0
    assert acc >= 0
