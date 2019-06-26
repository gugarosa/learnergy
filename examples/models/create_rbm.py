from recogners.models.rbm import RBM

# Creating an RBM
model = RBM(n_visible=128, n_hidden=128, steps=1, learning_rate=0.1, momentum=0.5, decay=0.001, emperature=1)

# Checking its parameters
print(model.W, model.a, model.b)
