import torch
import pickle

pickle.dumps(torch.randn(10).shape)  # TypeError: can't pickle Size objects