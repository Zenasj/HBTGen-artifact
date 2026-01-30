import torch

#  torch/tensor.py
with torch.no_grad():
   ...
   new_tensor = self.new()    # `at::GradMode` is false at this point
   ...