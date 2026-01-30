import torch
import pickle

@torch.no_grad()
def foo(x):
    return 3

pickle.dumps((foo,))