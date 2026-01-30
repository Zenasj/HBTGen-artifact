model = retinanet_resnet50_fpn_v2(pretrained=True)
device = torch.device('cuda')

model.to(device).eval()
model = torch.compile(model)
input_data = torch.randn(3, 512, 512)
with torch.no_grad():
    result = model([input_data.to(device)])

import torch
import torchvision
mod_ = torchvision.models.get_model('retinanet_resnet50_fpn_v2', pretrained=True)
mod_.to(device='cuda').eval()
mod = torch.compile(mod_)
with torch.no_grad():
    out = mod(torch.randn(1, 3, 224, 224, device='cuda'))