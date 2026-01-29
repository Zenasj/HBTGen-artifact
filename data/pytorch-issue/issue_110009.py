# torch.rand(5, 10, dtype=torch.float32)  # Inferred input shape for the matrix A

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, k=2):
        super(MyModel, self).__init__()
        self.k = k

    def forward(self, A):
        b = A.t() @ A
        u, d, v = torch.svd(b)
        
        # torch.linalg.pinv
        pinv = torch.linalg.pinv(b, rtol=(d[self.k] / d[0]).detach())
        loss_pinv = pinv.sum()
        
        # torch.svd
        d_new = torch.zeros_like(d)
        d_new[:self.k] = 1 / d[:self.k]
        pinv_svd = u @ torch.diag(d_new) @ v.t()
        loss_svd = pinv_svd.sum()
        
        # Return the difference in the losses
        return (loss_pinv - loss_svd).abs()

def my_model_function():
    return MyModel()

def GetInput():
    A = torch.rand(5, 10, requires_grad=True)
    return A

