import torch
import torchvision

# Load a segmentation model from torchvision
model = torchvision.models.segmentation.fcn_resnet101(pretrained=True, progress=True, num_classes=21)

example = torch.ones(1, 3, 224, 224) 

out = model(example)['out']

print(type(out), out.shape)

traced_script_module = torch.jit.script(model)

output = traced_script_module(example)['out']

print(type(output), output.shape)

print(out[0, 1, :, :], output[0, 1, :, :])