import torch
torch.linspace(1j, 2j, steps=100) # Unnecessary warning raised.
torch.linspace(1j, 2j, steps=100, dtype=torch.float) # Valid warning raised/future: throw an error.