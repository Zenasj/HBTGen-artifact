# torch.rand(10, 10, dtype=torch.float32, device='cuda')  # Input shape and dtype
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.stream = torch.cuda.Stream()  # Predefined stream for custom computation

    def forward(self, base):
        view = base[5:]  # Create a view with storage offset > 0
        with torch.cuda.stream(self.stream):
            torch.cuda._sleep(50000000)  # Simulate computation on the custom stream
        view.record_stream(self.stream)  # Record stream on the view tensor
        return view  # Ensure the view is used to maintain reference

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(10, 10, dtype=torch.float32, device='cuda')  # Matches input requirements

