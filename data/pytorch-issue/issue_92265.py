import torch
from torch.utils.data import TensorDataset, DataLoader

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS is available")
else:
    device = torch.device("cpu")
    print("MPS is not available")

inputs = torch.randn(3, 1, device=device)
targets = torch.tensor([1, 2, 3], device=device)

dataset = TensorDataset(inputs, targets)

dataloader = DataLoader(dataset, batch_size=1)

# Iterating over the dataloader
for i, (input, target) in enumerate(dataloader):
    print(
        f'DataLoader: batch {i} => target {target.item()} but should be {i+1}')