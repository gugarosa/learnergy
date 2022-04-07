import torch

from learnergy.models.gaussian import gaussian_rbm
 
def test_gaussian_rbm_normalize_setter():
    new_gaussian_rbm = gaussian_rbm.GaussianRBM()

    new_gaussian_rbm.normalize = False
    
    assert new_gaussian_rbm.normalize == False


def test_gaussian_rbm_normalize():
    new_gaussian_rbm = gaussian_rbm.GaussianRBM()

    assert new_gaussian_rbm.normalize == True

def test_gaussian_rbm_input_normalize_setter():
    new_gaussian_rbm = gaussian_rbm.GaussianRBM()

    new_gaussian_rbm.input_normalize = False 

    assert new_gaussian_rbm.input_normalize == False 


def test_gaussian_rbm_input_normalize():
    new_gaussian_rbm = gaussian_rbm.GaussianRBM()

    assert new_gaussian_rbm.input_normalize == True



def test_gaussian_rbm_visible_sampling():
    new_gaussian_rbm = gaussian_rbm.GaussianRBM()

    h = torch.ones(1, 128)

    probs, states = new_gaussian_rbm.visible_sampling(h, scale=True)

    assert probs.size(1) == 128
    assert states.size(1) == 128

    probs, states = new_gaussian_rbm.visible_sampling(h, scale=False)

    assert probs.size(1) == 128
    assert states.size(1) == 128


def test_gaussian_relu_rbm_hidden_sampling():
    new_gaussian_relu_rbm = gaussian_rbm.GaussianReluRBM()

    v = torch.ones(1, 128)

    probs, states = new_gaussian_relu_rbm.hidden_sampling(v, scale=True)

    assert probs.size(1) == 128
    assert states.size(1) == 128

    probs, states = new_gaussian_relu_rbm.hidden_sampling(v, scale=False)

    assert probs.size(1) == 128
    assert states.size(1) == 128

def test_gaussian_selu_rbm_hidden_sampling():
    new_gaussian_selu_rbm = gaussian_rbm.GaussianSeluRBM()

    v = torch.ones(1, 128)

    probs, states = new_gaussian_selu_rbm.hidden_sampling(v, scale=True)

    assert probs.size(1) == 128
    assert states.size(1) == 128

    probs, states = new_gaussian_selu_rbm.hidden_sampling(v, scale=False)

    assert probs.size(1) == 128
    assert states.size(1) == 128


def test_variance_gaussian_rbm_sigma():
    new_variance_gaussian_rbm = gaussian_rbm.VarianceGaussianRBM()

    assert new_variance_gaussian_rbm.sigma.size(0) == 128


def test_variance_gaussian_rbm_sigma_setter():
    new_variance_gaussian_rbm = gaussian_rbm.VarianceGaussianRBM()

    try:
        new_variance_gaussian_rbm.sigma = 1
    except:
        new_variance_gaussian_rbm.sigma = torch.nn.Parameter(torch.zeros(128))

    assert new_variance_gaussian_rbm.sigma.size(0) == 128


def test_variance_gaussian_rbm_hidden_sampling():
    new_variance_gaussian_rbm = gaussian_rbm.VarianceGaussianRBM()

    v = torch.ones(1, 128)

    probs, states = new_variance_gaussian_rbm.hidden_sampling(v, scale=True)

    assert probs.size(1) == 128
    assert states.size(1) == 128

    probs, states = new_variance_gaussian_rbm.hidden_sampling(v, scale=False)

    assert probs.size(1) == 128
    assert states.size(1) == 128


def test_variance_gaussian_rbm_visible_sampling():
    new_variance_gaussian_rbm = gaussian_rbm.VarianceGaussianRBM()

    h = torch.ones(1, 128)

    probs, states = new_variance_gaussian_rbm.visible_sampling(h, scale=True)

    assert probs.size(1) == 128
    assert states.size(1) == 128

    probs, states = new_variance_gaussian_rbm.visible_sampling(h, scale=False)

    assert probs.size(1) == 128
    assert states.size(1) == 128


def test_variance_gaussian_rbm_energy():
    new_variance_gaussian_rbm = gaussian_rbm.VarianceGaussianRBM()

    samples = torch.ones(1, 128)

    energy = new_variance_gaussian_rbm.energy(samples)

    assert energy.detach().numpy() < 0
