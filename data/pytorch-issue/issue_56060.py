# torch.rand(1, 3, 224, 224, dtype=torch.float32)
import torch
import torchvision
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Load the original model and set to eval mode (critical fix from comments)
        self.original = torchvision.models.segmentation.fcn_resnet101(pretrained=True, num_classes=21)
        self.original.eval()
        # Create TorchScript version of the same model (must be done in eval mode)
        self.scripted = torch.jit.script(self.original)

    def forward(self, x):
        # Run both models and compare outputs
        orig_out = self.original(x)['out']
        script_out = self.scripted(x)['out']
        # Check numerical equivalence with tolerance (addresses original discrepancy)
        are_close = torch.allclose(orig_out, script_out, atol=1e-5, rtol=1e-5)
        return torch.tensor([are_close], dtype=torch.bool)  # Return boolean as tensor

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

