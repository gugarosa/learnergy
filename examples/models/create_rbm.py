from recogners.models.rbm import RBM

# Creating an RBM
r = RBM(n_visible=128, n_hidden=128, learning_rate=0.1, steps=1, temperature=1)

# Checking its parameters
print(r.W, r.a, r.b)
