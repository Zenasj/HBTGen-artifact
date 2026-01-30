import torch.nn as nn
import torchvision

import torch
from torchvision.models import vgg16_bn
model = vgg16_bn()
model.eval()
example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save('vgg16_bn_float32.pt')
model.qconfig = torch.quantization.default_qconfig
torch.quantization.prepare(model, inplace=True)
quantized=torch.quantization.convert(model, inplace=True)
example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(quantized, example)
traced_script_module.save('vgg16_bn_qint8.pth')

import torch
from torchvision.models import vgg16_bn

def uninplace(model):
    """Sets all `inplace` values to False"""
    if hasattr(model, 'inplace'):
        model.inplace = False
    if not model.children():
        return
    for child in model.children():
        uninplace(child)

def prep_for_fusion(model, parent_name):
    """Fuses all conv+bn+relu, conv+bn, and conv+relu"""
    if not model.children():
        return []
    result = []
    candidate = []
    for name, child in model.named_children():
        new_name = parent_name + '.' + name
        if new_name[0] == '.':
            new_name = new_name[1:]
        if type(child) == torch.nn.Sequential:
            candidate = []
            result.extend(prep_for_fusion(child, new_name))
        else:
            if len(candidate) == 0 and type(child) == torch.nn.Conv2d:
                candidate = [new_name]
            elif len(candidate) == 1 and type(child) == torch.nn.ReLU:
                candidate.append(new_name)
                result.append(candidate)
                candidate = []
            elif len(candidate) == 1 and type(child) == torch.nn.BatchNorm2d:
                candidate.append(new_name)
            elif len(candidate) == 2:
                if type(child) == torch.nn.ReLU:
                    candidate.append(new_name)
                result.append(candidate)
                candidate = []
    return result

model = vgg16_bn()
model.eval()

example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save('vgg16_bn_float32.pt')

uninplace(model)  # This is a hack!
modules_to_fuse = prep_for_fusion(model, '')

model.qconfig = torch.quantization.default_qconfig
fused_model = torch.quantization.fuse_modules(model, modules_to_fuse)

torch.quantization.prepare(fused_model, inplace=True)
quantized = torch.quantization.convert(fused_model, inplace=False)

q_example = torch.quantize_per_tensor(example, scale=1e-3, zero_point=128,
                                      dtype=torch.quint8)


traced_script_module = torch.jit.trace(quantized, q_example, check_trace=False)
traced_script_module.save('vgg16_bn_qint8.pth')

COMPLEX_MODULES = (nn.Sequential,)  # Add any "recursable" modules in here

def prep_for_fusion(model, parent_name=''):
  """Fuses all conv+bn+relu, conv+bn, and conv+relu"""
  if not model.children():
    return []
  result = []
  candidate = []
  for name, child in model.named_children():
    new_name = parent_name + '.' + name
    if new_name[0] == '.':
      new_name = new_name[1:]
    if type(child) in COMPLEX_MODULES:
      candidate = []
      result.extend(prep_for_fusion(child, new_name))
    else:
      if type(child) not in (nn.Conv2d, nn.BatchNorm2d, nn.ReLU):
        # Not fusable layer found, check if candidate exists, and drop it
        if len(candidate) > 1:
          result.append(candidate)
        candidate = []
        continue
      if len(candidate) == 0 and type(child) == nn.Conv2d:
        # No candidates, so starting a new one
        candidate = [new_name]
      elif len(candidate) > 0 and type(child) == nn.ReLU:
        # There is a candidate, and we are done
        candidate.append(new_name)
        result.append(candidate)
        candidate = []
      elif len(candidate) > 0 and type(child) == nn.BatchNorm2d:
        # So far Conv + Bn found, waiting for ReLU?
        candidate.append(new_name)
  return result

modules_to_fuse = prep_for_fusion(model)

COMPLEX_MODULES = (nn.Sequential, torchvision.models.resnet.BasicBlock, torchvision.models.resnet.Bottleneck)