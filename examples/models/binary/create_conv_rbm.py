from learnergy.models.binary import ConvRBM

# Creates a ConvRBM-based class
model = ConvRBM(visible_shape=(28, 28), filter_shape=(7, 7), n_filters=10, n_channels=1,
                steps=1, learning_rate=0.01, momentum=0, decay=0, use_gpu=True)