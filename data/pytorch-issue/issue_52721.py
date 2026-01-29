# torch.rand(1, dtype=torch.float32)  # Dummy scalar input
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        success = 0.0
        
        # Test 1: Scalar tensor creation after __getitem__ wrapping
        a = torch.tensor(1)
        try:
            old_getitem = torch.Tensor.__getitem__
            def wrapper(func):
                def new_func(*args, **kwargs):
                    print('hi')
                    return func(*args, **kwargs)
                return new_func
            torch.Tensor.__getitem__ = wrapper(old_getitem)
            # This should raise TypeError
            _ = torch.tensor([a])
        except TypeError:
            success += 1.0
        finally:
            torch.Tensor.__getitem__ = old_getitem

        # Test 2: Tensor indexing after __getitem__ wrapping
        a = torch.ones((10, 1000))
        b = torch.tensor([1, 2, 3], dtype=torch.int64)
        try:
            old_getitem = torch.Tensor.__getitem__
            torch.Tensor.__getitem__ = wrapper(old_getitem)
            # This should raise IndexError
            _ = a[b]
        except IndexError:
            success += 1.0
        finally:
            torch.Tensor.__getitem__ = old_getitem

        return torch.tensor(1.0) if success == 2.0 else torch.tensor(0.0)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, dtype=torch.float32)

