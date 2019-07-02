def unitary(x, eps=1e-8):
    """
    """

    print(x)

    #
    x1 = x.copy()

    #
    x1 -= x.min()

    #
    x1 *= 1.0 / (x.max() + eps)

    print(f'\n{x1}')

    return x1