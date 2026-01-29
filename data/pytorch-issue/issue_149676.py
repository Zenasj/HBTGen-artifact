# torch.rand(1, dtype=torch.float32)  # Dummy input (not used in computation)
import torch
from torch import nn

class CtxMgr:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

class MyModel(nn.Module):
    def forward(self, x):
        with CtxMgr():
            with CtxMgr():
                pass
            with CtxMgr():
                with CtxMgr():
                    pass
                torch._dynamo.graph_break()
        return x  # Return input tensor as output (no computation)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, dtype=torch.float32)

