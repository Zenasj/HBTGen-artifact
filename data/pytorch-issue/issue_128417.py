import torch
import torch.nn as nn
import torchvision

class MyPreprocess(nn.Module):
    def forward(self, img):
        w = max(img.shape[1], img.shape[2])
        return torchvision.transforms.v2.CenterCrop(size = (w, w))(img)
pm = MyPreprocess()
inputs = (imag,)
ex_orogram = torch.export.export(pm, inputs)