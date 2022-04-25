from learnergy.models.deep import ConvDBN

# Creates a ConvDBN-based class
model = ConvDBN(
    model="bernoulli",
    visible_shape=(28, 28),
    filter_shape=((2, 2), (2, 2)),
    n_filters=(4, 4),
    steps=(1, 1),
    n_channels=1,
    learning_rate=(0.1, 0.1),
    momentum=(0, 0),
    decay=(0, 0),
    use_gpu=True,
)
