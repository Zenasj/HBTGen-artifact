import torch

loss = criterion(pred, target) # target comes from Dataset and is of type torch.DoubleTensor,
                               # but pred is of type torch.Tensor
loss.backward()