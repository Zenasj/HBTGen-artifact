import torchvision

import torch

torch.use_deterministic_algorithms(True)

from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn

model = fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False).cuda()

model(
    torch.zeros(1, 3, 800, 800).cuda(), [{
        'boxes': torch.tensor([[100, 200, 300, 400]]).cuda(), 
        'labels': torch.tensor([1]).cuda(),
    }])

x[torch.tensor([1, 3]).cuda()] = torch.tensor([2,2]).cuda()