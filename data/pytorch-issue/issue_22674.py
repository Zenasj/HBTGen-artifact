import torch
import torchvision
import torch.onnx
from torch.autograd import  Variable

# An instance of your model
model = torchvision.models.mobilenet_v2(pretrained=True)
print(model)

x = torch.randn(1, 3, 224, 224,requires_grad=True)
# Export the model
torch_out = torch.onnx._export(model, x, "mobilenet_v2.onnx", export_params=True)