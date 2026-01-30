import torch
import torch.nn as nn
        
class DummyTensor(torch.Tensor):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 2)
    def forward(self, x):
        return DummyTensor(self.fc(x))
    
data = torch.Tensor([[1, 2],[3, 4],[5, 6]])
model = SimpleModel()
before_hook_output = model(data)

def backward_hook(module, grad_input, grad_output):
    return #do nothing
model.register_full_backward_hook(backward_hook)

after_hook_output = model(data)

print("Model output type before applying wrapper: ", type(before_hook_output))
print("Model output type after applying wrapper: ", type(after_hook_output))