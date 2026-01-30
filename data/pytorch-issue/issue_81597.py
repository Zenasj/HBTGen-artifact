import torch

# torch.Tensors cannot be used as a key in a dictionary
# because they define a custom __eq__ function which when used
# to resolve hash collisions will throw when comparing tensors:
# "RuntimeError: bool value of Tensor with more than one value is ambiguous."