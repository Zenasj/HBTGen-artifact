import torch

x = torch.eye(2)
print(x.std(), x.to("mps").std())
# tensor(0.5774) tensor(0.5000, device='mps:0')