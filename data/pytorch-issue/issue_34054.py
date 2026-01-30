import torch
import torch.nn as nn
import onnxruntime

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
    
    def forward(self, x):
        x = x.clone()
        x[x <= 1] = 0
        x[x > 1] = 1
        return x

model = Net()
x=torch.tensor([4, 1, 3])
torch.onnx.export(model, x, "dummy.pt", verbose=True, opset_version=11)