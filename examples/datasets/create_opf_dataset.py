from torch.utils.data import DataLoader

from recogners.datasets.opf import OPFDataset

# Declaring OPF file format to be loaded
opf_file = ''

# Creating the OPF dataset object
d = OPFDataset(path=opf_file)

# Creating a PyTorch's generator
g = DataLoader(d, batch_size=16, shuffle=True, num_workers=1)
