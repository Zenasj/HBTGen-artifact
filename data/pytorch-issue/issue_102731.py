# torch.rand(2, 2, dtype=torch.float32)
import copy
import torch
import torch.nn as nn

class ToyModel(nn.Module):
    def __init__(self, data_preprocessor=None):
        super().__init__()
        self.linear1 = nn.Linear(2, 2)
        self.norm = nn.BatchNorm1d(2)
        self.linear2 = nn.Linear(2, 1)

    def forward(self, inputs):
        if isinstance(inputs, list):
            inputs = torch.stack(inputs)
        outputs = self.linear1(inputs)
        outputs = self.norm(outputs)
        outputs = self.linear2(outputs)
        return outputs

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ddp_model = ToyModel()
        self.fsdp_model = copy.deepcopy(self.ddp_model)  # Ensure initial weights/buffers match

    def forward(self, x):
        # Simulate AMP context and loss computation as per original issue
        with torch.cuda.amp.autocast():
            ddp_out = self.ddp_model(x)
            fsdp_out = self.fsdp_model(x)
            ddp_loss = ddp_out.sum()
            fsdp_loss = fsdp_out.sum()
        # Return 1.0 if losses differ beyond tolerance, else 0.0
        return torch.tensor(1.0 if not torch.allclose(ddp_loss, fsdp_loss, atol=1e-5) else 0.0, dtype=torch.float32)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 2, dtype=torch.float32)

