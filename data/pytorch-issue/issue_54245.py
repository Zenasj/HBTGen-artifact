import torchvision

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn


def main():
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    init_img = torch.zeros((1, 3, 512, 512), device=device)
    target = [{"boxes": torch.as_tensor([[1, 2, 3, 4]], device=device),
               "labels": torch.as_tensor([1], device=device)}]
    model = fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False).to(device)
    model.train()
    model(init_img, target)


if __name__ == '__main__':
    main()

import torch
torch.randperm(159826, device='cuda')