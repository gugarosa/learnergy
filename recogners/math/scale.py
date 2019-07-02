def unitary(x, eps=1e-8):
    """Scales an array between 0 and 1.

    Args:
        x (array): A numpy array to be scaled.
        eps (float): An epsilon value to avoid division by zero.

    Returns:
        The scaled array.

    """

    # Makes sure the array is double typed
    x = x.astype('float64')

    # Gathers array minimum and subtract
    x -= x.min()

    # Normalizes the array using its maximum
    x *= 1.0 / (x.max() + eps)

    return x
