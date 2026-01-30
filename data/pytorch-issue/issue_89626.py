import pickle
import torch
from torch.fx.experimental.proxy_tensor import make_fx

inputs = [torch.randn(2, 3)]
def forward(x):
    return torch.tanh(x)

pickle.dumps(inputs)
print("Pickling works OK before make_fx")
make_fx(forward)(*inputs)
pickle.dumps(inputs) # Throws exception