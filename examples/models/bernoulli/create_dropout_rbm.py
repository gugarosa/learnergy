from learnergy.models.bernoulli import DropoutRBM

# Creates a DropoutRBM-based class
model = DropoutRBM(n_visible=784, n_hidden=128, steps=1, learning_rate=0.1,
                   momentum=0, decay=0, temperature=1, dropout=0.5, use_gpu=False)
