import torch.onnx
import torch.nn as nn

class test_torch(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x= torch.minimum(x, x)
        return x

net = test_torch()
test_input = torch.randn(1, 2, 224, 224) 
torch.onnx.export(net.to('cpu'), test_input,'convnet.onnx')