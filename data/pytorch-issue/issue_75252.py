import torch
import torch.nn as nn
import torchvision.models as models
from openvino.inference_engine import IECore

model = models.resnet50(pretrained=True)
model.eval()
dummy_input = torch.randn(1, 3, 224, 224)
input_names = [ "actual_input" ]
output_names = [ "output" ]
torch.onnx.export(model, 
              dummy_input,
              "resnet50.onnx",
              verbose=False,
              input_names=input_names,
              output_names=output_names,
              export_params=True,
              )
ie = IECore()
net = ie.read_network('resnet50.onnx')
model = ie.load_network(network=net, device_name="CPU")

import torch
import torch.nn as nn
import torchvision.models as models
from openvino.inference_engine import IECore

model = models.resnet50(pretrained=True)
model.eval()

for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.track_running_stats = False
            module.running_var = None
            module.running_mean = None
            
dummy_input = torch.randn(1, 3, 224, 224)
input_names = [ "actual_input" ]
output_names = [ "output" ]
torch.onnx.export(model, 
              dummy_input,
              "resnet50.onnx",
              verbose=False,
              input_names=input_names,
              output_names=output_names,
              export_params=True,
              )
ie = IECore()
net = ie.read_network('resnet50.onnx')
model = ie.load_network(network=net, device_name="CPU")