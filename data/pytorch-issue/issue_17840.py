# torch.rand(1, 3, 64, 64, dtype=torch.float32)
import torch
import torch.nn as nn
from torchvision.models import resnet34

class OutputHook:
    "Create a hook on `m` with to store the output."
    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)
        self.removed = False

    def hook_fn(self, module, input, output):
        "Applies `hook_func` to `module`, `input`, `output`."
        self.stored = output

    def remove(self):
        "Remove the hook from the model."
        if not self.removed:
            self.hook.remove()
            self.removed = True

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet34(pretrained=True)
        # Register hooks on all children modules (root cause of thread deadlock)
        self.hooks = [OutputHook(child) for child in self.model.children()]
        self.model.eval()  # Matches original eval() context in the issue's example

    def forward(self, x):
        return self.model(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 64, 64, dtype=torch.float32)

