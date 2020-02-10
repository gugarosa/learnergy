def unitary_scale(x, eps=1e-10):
    """Scales an array between 0 and 1.

    Args:
        x (array): A numpy array to be scaled.
        eps (float): An epsilon value to avoid division by zero.

    Returns:
        The scaled array.

    """

    # Makes sure the array is float typed
    x = x.astype('float32')

    # Gathers array minimum and subtract
    x -= x.min()

    # Normalizes the array using its maximum
    x *= 1.0 / (x.max() + eps)

    return x
