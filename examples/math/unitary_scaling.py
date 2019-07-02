import recogners.math.scale as scale
import numpy as np

# Creates an input array
a = np.array([1, 2, 3, 4, 5])

# Scales the input array between 0 and 1
u = scale.unitary(a)

# Prints the result
print(u)