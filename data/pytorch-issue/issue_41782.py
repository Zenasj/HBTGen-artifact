# test.py
import torch
print(f"Torch version: {torch.__version__}")

x = torch.tensor(data=[[1,2],[3,4]], dtype=torch.long, device=None)
x = x.to('cuda:0')
seed = torch.seed()