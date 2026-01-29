# torch.rand(3, dtype=torch.float32)  # Input shape is 1D tensor of size 3
import torch
import torch.nn as nn

class MyModel(nn.Module):
    class Subclass(torch.Tensor):
        pass  # Minimal Tensor subclass without __torch_function__ overrides

    def __init__(self):
        super().__init__()
        # Required to enable Dynamo tracing for the Tensor subclass
        import torch._dynamo
        torch._dynamo.config.traceable_tensor_subclasses.add(MyModel.Subclass)

    def forward(self, x):
        # Wrap input tensor in the custom subclass
        subclass_x = MyModel.Subclass(x)
        return torch.max(torch.abs(subclass_x))

def my_model_function():
    # Returns initialized model instance with Dynamo configuration
    return MyModel()

def GetInput():
    # Returns random tensor matching expected input shape (3,)
    return torch.rand(3, dtype=torch.float32)

