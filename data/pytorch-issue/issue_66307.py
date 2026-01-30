import torch
import torchvision

import torch.utils.mobile_optimizer as mobile_optimizer
from torchvision.models.segmentation import fcn_resnet50, fcn_resnet101, \
    deeplabv3_resnet50, deeplabv3_resnet101, deeplabv3_mobilenet_v3_large, lraspp_mobilenet_v3_large

model = lraspp_mobilenet_v3_large(pretrained=True)
model.eval()
script_model = torch.jit.script(model)
optimized_model = mobile_optimizer.optimize_for_mobile(script_model, backend='Metal')
print(torch.jit.export_opnames(optimized_model))