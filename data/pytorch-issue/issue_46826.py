from torch import Tensor
import torch

class MyTensor(Tensor):
    pass

a = MyTensor([1,2,3])
b = Tensor([1,2,3])

type(a.shape), type(b.shape)

(torch.Size, torch.Size)

(tuple, torch.Size)

class MyTensor(Tensor):
    @property
    def shape(self):
        return torch.Size(super(Tensor, self).shape)