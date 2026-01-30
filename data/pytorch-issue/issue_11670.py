import torch
import torch.nn as nn
import torch.onnx as torch_onnx
from onnx_coreml import convert

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), stride=1, padding=0, bias=True)

    def forward(self, inputs):
        x = self.conv(inputs)
        # The view create the reshape/unsqueeze issue when exporting to coreml
        x = x.view(x.size()[0], -1)
        #x = x.view(1, 307328)
        print(x.size())
        return x

# Use this an input trace to serialize the model
input_shape = (3, 100, 100)
model_onnx_path = "torch_model.onnx"
model = Model()
model.train(False)

# Export the model to an ONNX file
dummy_input = torch.randn(1, *input_shape)
output = torch_onnx.export(model, 
                          dummy_input, 
                          model_onnx_path, 
                          export_params=True,
                          verbose=True)

# Export to coreml
coreml_model = convert('./torch_model.onnx')
coreml_model.save('./test.mlmodel')