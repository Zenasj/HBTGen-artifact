# torch.rand(1, dtype=torch.float32, device='cuda')  # Dummy input tensor (shape arbitrary, not used in computation)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.t = nn.Parameter(torch.tensor([1., 2.], requires_grad=True, device='cuda'))
        
        def callback():
            print("Callback! Raising an error...")
            raise RuntimeError("Error from callback!")
        
        def hook_with_callback(*args):
            print("Backward hook!")
            torch.autograd.Variable._execution_engine.queue_callback(callback)
        
        self.t.register_hook(hook_with_callback)
    
    def forward(self, x):
        # x is a dummy input (not used in computation)
        return self.t ** 2

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, device='cuda')

