import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile

model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
model.eval()

traced_script_module = torch.jit.script(model)
traced_script_module_optimized = optimize_for_mobile(traced_script_module)