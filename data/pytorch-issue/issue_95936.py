import torch

try:
    self.model = self.model.to("gpu")
except torch.cuda.OutOfMemoryError: 
    self.model = self.model.to("cpu")