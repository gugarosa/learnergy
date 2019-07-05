from recogners.models.rbm import RBM

# Creating an RBM
model = RBM()

# Loading an RBM from saved model
model.load('rbm.pkl')

# Checking its parameters
print(model.W, model.a, model.b)
