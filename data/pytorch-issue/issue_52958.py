# torch.rand(B, C, H, W, dtype=torch.float32)  # Inferred input shape: (batch_size, channels, height, width)

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 1, 3, 1, padding=1)

    def forward(self, img):
        stride = (85, 85)
        crop_size = (128, 128)

        c = img.size(1)
        h = img.size(2)
        w = img.size(3)

        pad_h = (crop_size[0] - h) % crop_size[0]
        pad_w = (crop_size[1] - w) % crop_size[1]

        img = F.pad(img, (0, pad_w, 0, pad_h))
        unfold = F.unfold(img, crop_size, stride=stride)

        win_count = unfold.size(2)

        imgs = unfold.permute([0, 2, 1])
        imgs = torch.reshape(imgs, (-1, c, *crop_size))

        preds = []
        split_imgs = torch.split(imgs, 2, 0)
        for img in split_imgs:
            preds.append(self.c1(img))

        pred = torch.cat(preds, 0)
        pred = torch.reshape(pred, (-1, win_count, crop_size[0] * crop_size[1]))
        pred = pred.permute([0, 2, 1])

        # Currently, ONNX export does not support torch.fold
        # pred = torch.fold(pred, [h, w], crop_size, stride=stride)

        return pred

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    batch_size = 2
    channels = 3
    height = 224
    width = 224
    return torch.rand(batch_size, channels, height, width, dtype=torch.float32)

