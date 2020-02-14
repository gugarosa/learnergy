from learnergy.models.dbn import DBN

# Creates a DBN-based class
model = DBN(n_visible=784, n_hidden=[128, 256, 512], steps=1, learning_rate=0.1,
            momentum=0, decay=0, temperature=1, use_gpu=False)
