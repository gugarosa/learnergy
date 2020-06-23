from learnergy.models.real import SigmoidRBM

# Creates a SigmoidRBM-based class
model = SigmoidRBM(n_visible=784, n_hidden=128, steps=1, learning_rate=0.1,
                   momentum=0, decay=0, temperature=1, use_gpu=False)
