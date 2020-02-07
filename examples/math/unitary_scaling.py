import numpy as np

import learnergy.math.scale as s

# Creates an input array
a = np.array([1, 2, 3, 4, 5])

# Scales the input array between 0 and 1
u = s.unitary_scale(a)

# Prints the result
print(u)
