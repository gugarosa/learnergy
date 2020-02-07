from torch.utils.data import DataLoader

from learnergy.datasets.opf import OPFDataset

# Declaring OPF file format to be loaded
opf_file = ''

# Creating the OPF dataset object
dataset = OPFDataset(path=opf_file)

# Creating a PyTorch's generator
batches = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=1)
