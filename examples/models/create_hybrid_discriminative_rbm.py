from learnergy.models.discriminative_rbm import HybridDiscriminativeRBM

# Creates a HybridDiscriminativeRBM-based class
model = HybridDiscriminativeRBM(n_visible=784, n_hidden=128, learning_rate=0.1,
                                alpha=0.01, momentum=0, decay=0, use_gpu=False)
