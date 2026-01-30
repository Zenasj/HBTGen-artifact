import torch.nn as nn
import torch
from torch.quantization import get_default_qconfig
from torch.quantization.quantize_fx import prepare_fx, convert_fx

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=1, 
                    out_channels=64, 
                    kernel_size=(8, 20), 
                    stride=(1, 1),
                    padding=0)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(num_features=64)

    def forward(self, x):
        # x = [batch_size, channel=1, height=10, width=20]
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        return x
    
model = MyModel()

## Quantize model (post-training Static quantization FX Graph mode)
torch.backends.quantized.engine = "qnnpack"

model.to("cpu")
model.eval()

prepared_model = prepare_fx(model, {"": get_default_qconfig("qnnpack")})

with torch.no_grad(): # calibrate using random data
    data = torch.rand((7,1,10,20))
    prepared_model(data)
model_quantized = convert_fx(prepared_model)

## Prediction using scripted model an error
model_quantized_scripted = torch.jit.script(model_quantized) # script model
model_quantized_scripted(torch.rand((7,1,10,20))) # raises an error

from torch.ao.quantization import get_default_qconfig_mapping
torch.backends.quantized.engine = "x86"
prepared_model = prepare_fx(model, get_default_qconfig_mapping("x86"), example_inputs=torch.rand((7,1,10,20)))