from learnergy.models.residual_dbn import ResidualDBN

# Creates a ResidualDBN-based class
model = ResidualDBN(model='bernoulli', n_visible=784, n_hidden=[128, 256, 128], steps=[1, 1, 1],
            learning_rate=[0.1, 0.1, 0.1], momentum=[0, 0, 0], decay=[0, 0, 0], temperature=[1, 1, 1],
            alpha=1, beta=1, use_gpu=True)
