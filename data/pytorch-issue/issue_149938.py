# torch.rand(B, C, dtype=torch.float32)
import torch
import torch.nn as nn

batch_size = 1
in_features = 4
out_features = 8
device = 'cpu'
factory_kwargs = {'device': device, 'dtype': torch.float32}

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False, **factory_kwargs)
        self.grad_available = []  # Stores gradient availability for each backward path
        
        def hook(module, grad_input, grad_output):
            # Track whether parameter gradients are available during hook execution
            param = next(module.parameters())
            grad_available = param.grad is not None
            self.grad_available.append(grad_available)
            
        self.fc.register_full_backward_hook(hook)
    
    def forward(self, input):
        # Split input into two paths with/without requires_grad
        input_with_grad = input.detach().requires_grad_(True)
        input_without_grad = input.detach().requires_grad_(False)
        
        # Process both paths through the same module
        output_with = self.fc(input_with_grad)
        output_without = self.fc(input_without_grad)
        
        return output_with, output_without

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(batch_size, in_features, **factory_kwargs)

