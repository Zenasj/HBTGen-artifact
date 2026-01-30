import torch.nn as nn

import torch

class Model(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, image):
        batch_size, channels, height, width = image.shape
        x = torch.arange(width)
        y = torch.arange(height)
        X,Y = torch.meshgrid(x, y, indexing='xy')
        return X, Y
    
image = torch.zeros(4,3,192,256)
model = Model()
torch.onnx.export(model, image, '/tmp/model.onnx')

import torch

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, image):
        batch_size, channels, height, width = image.shape
        x = torch.arange(width)
        y = torch.arange(height)
        X,Y = torch.meshgrid(x, y, indexing='xy')
        return X, Y

image = torch.zeros(4,3,192,256)
model = Model()
onnx_program = torch.onnx.export(model, (image,), dynamo=True, report=True)
print(onnx_program)