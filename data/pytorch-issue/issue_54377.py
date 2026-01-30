import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile

model = torchvision.models.mobilenet_v2(pretrained=True)
scripted_model = torch.jit.script(model)
optimized_model = optimize_for_mobile(scripted_model, backend='metal') # <<< error here
print(torch.jit.export_opnames(optimized_model))
torch.jit.save(optimized_model, './mobilenetv2_metal.pt')

model = torchvision.models.mobilenet_v2(pretrained=True)
scripted_model = torch.jit.script(model)
torch.jit.save(scripted_model, './mobilenetv2.pt')

model = torch.jit.load('mobilenetv2.pt')
optimized_model = optimize_for_mobile(model, backend="metal")
print(torch.jit.export_opnames(optimized_model))
torch.jit.save(optimized_model, './mobilenetv2_metal.pt')