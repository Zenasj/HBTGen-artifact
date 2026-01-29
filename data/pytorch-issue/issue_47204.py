# torch.rand(12)  # Input shape is a 1D tensor of length 12
import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_LEN = 12
NUM_GESTURES = 6

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.fc1 = nn.Linear(INPUT_LEN, 8)
        self.fc2 = nn.Linear(8, 4)
        self.fc3 = nn.Linear(4, NUM_GESTURES)
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = x.view(-1, INPUT_LEN)  # Critical reshape for ONNX export failure
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        h = self.dequant(h)
        return h

def my_model_function():
    # Returns a prepared quantization model to reproduce export error
    model = MyModel()
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    torch.quantization.prepare(model, inplace=True)  # Must be prepared but not calibrated
    return model

def GetInput():
    return torch.rand(INPUT_LEN)  # Matches 1D input expectation

