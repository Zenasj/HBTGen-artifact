import torch.nn as nn

import torch
class test_module(torch.nn.Module):
    def __init__(self):
        super(test_module, self).__init__()
        self.bn = torch.nn.BatchNorm1d(256, affine=False)
    def forward(self, x):
        B,N,D = x.shape
        x = x.view(-1, D)
        x = self.bn(x)
        x = x.view(B,N,D)
        return x
        
dummy_x= torch.randn(8,40,256)
net = test_module()
torch.onnx.export(net, (dummy_x), 
                "tmp.onnx", 
                input_names=['input'], 
                output_names=['preds'],
                verbose=True,
                opset_version=9)