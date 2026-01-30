import torchvision

import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
model = maskrcnn_resnet50_fpn()
# The following modifications cause the last command to fail
model.roi_heads.mask_roi_pool = None
model.roi_heads.mask_head = None
model.roi_heads.mask_predictor = None
model_script = torch.jit.script(model)