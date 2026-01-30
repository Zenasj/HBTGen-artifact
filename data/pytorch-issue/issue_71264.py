import torch
from torchvision import models
from torch.utils.mobile_optimizer import optimize_for_mobile
model = models.mobilenet_v2(pretrained=True)
scripted_model=torch.jit.script(model)
opt_m=optimize_for_mobile(scripted_model,backend='vulkan')
opt_m._save_for_lite_interpreter("mobilenet_v2_vulkan.ptl")

mModule = LiteModuleLoader.load(MainActivity.assetFilePath(getApplicationContext(), "mobilenet_v2_vulkan.ptl"),null,Device.VULKAN);

import torch
from torchvision import models
from torch.utils.mobile_optimizer import optimize_for_mobile
model = models.mobilenet_v2(pretrained=True)
scripted_model=torch.jit.script(model)
opt_m=optimize_for_mobile(scripted_model,backend='vulkan')
torch.jit.save(opt_m, "mobilenet_v2_vulkan.pt")
opt_m._save_for_lite_interpreter("mobilenet_v2_vulkan.ptl")

Module.load(MainActivity.assetFilePath(getApplicationContext(), "mobilenet_v2_vulkan.pt"),null, Device.VULKAN);