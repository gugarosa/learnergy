from learnergy.models.bernoulli import DiscriminativeRBM

# Creates a DiscriminativeRBM-based class
model = DiscriminativeRBM(
    n_visible=784, n_hidden=128, learning_rate=0.1, momentum=0, decay=0, use_gpu=False
)
