from learnergy.models.gaussian import GaussianConvRBM

# Creates a GaussianConvRBM-based class
model = GaussianConvRBM(
    visible_shape=(32, 32),
    filter_shape=(9, 9),
    n_filters=16,
    n_channels=3,
    steps=1,
    learning_rate=0.00001,
    momentum=0.5,
    decay=0,
    use_gpu=True,
)
