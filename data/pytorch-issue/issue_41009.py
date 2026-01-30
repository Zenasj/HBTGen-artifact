python
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx

# Model Architecture with Interpolate function
class MyModel(nn.Module):
    def __init__(self,scale_factor,mode):
        super(MyModel, self).__init__()
        self.interp_mode = mode
        self.sf = scale_factor

    def forward(self,x):
        x = F.interpolate(x,
            scale_factor=self.sf,
            mode=self.interp_mode)

        return x

#initialize model object
scale_factor = 1/2
mode = 'area'
model = MyModel(scale_factor,mode)

#export model to onnx
x = torch.randn(1,1,100,100)
model_out = model(x)
torch.onnx.export(model,
                  x,
                  'sample.onnx',
                  export_params=True,
                  opset_version=11,
                  input_names = ['input'],
                  output_names = ['output'])