import random

import sys
import os
import numpy as np
import onnxruntime as ort

import torch
import torch.nn as nn
from torch.autograd import Variable

# toy model with convtranspose3d
class PytorchModel(nn.Module):
    def __init__(self):
        super(PytorchModel, self).__init__()
        self.bn = nn.Sequential(
        	# nn.BatchNorm2d(3, momentum=0.2),
            nn.ConvTranspose3d(3, 3, (2,2,2), (2,2,2)),
        	)

    def forward(self, x):
        x = self.bn(x)
        return x

pytorch_model = PytorchModel()
pytorch_model.eval()

# export onnx
x_data = np.random.uniform(0, 1, size=(1, 3, 224, 224, 224))
x = Variable(torch.Tensor(x_data))
torch_out = torch.onnx._export(pytorch_model, x, "model.onnx", export_params=True)

# torch output
x_data = np.random.uniform(0, 1, size=(1, 3, 224, 224, 224))
x = Variable(torch.Tensor(x_data))
output_t = pytorch_model.cuda()(x.cuda())

model_path = 'model.onnx'
sess = ort.InferenceSession(model_path, None)
input_name = sess.get_inputs()[0].name  
output_name = sess.get_outputs()[0].name 

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# [BUG HERE] <- onnxruntime output  
output_o,  = sess.run([output_name], {input_name: to_numpy(x)})  

# test
print(output_o.shape)
print(np.testing.assert_allclose(to_numpy(output_t), output_o, rtol=1e-03, atol=1e-05))