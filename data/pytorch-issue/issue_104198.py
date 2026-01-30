import torch
import pickle

a, b = torch.arange(3), torch.arange(5) + 3
nt = torch.nested.nested_tensor([a, b])
with open('tensor.pkl', 'wb') as f:
    pickle.dump(nt, f)