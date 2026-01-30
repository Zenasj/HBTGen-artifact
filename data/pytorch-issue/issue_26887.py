import onnx
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Conv2d
from torch.nn import BatchNorm2d
import caffe2.python.onnx.backend as backend


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = Conv2d(3,
                            64,
                            kernel_size=7,
                            stride=2,
                            padding=3,
                            bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    dilation=1,
                                    ceil_mode=False)
        self.layer1 = Conv2d(64,
                            256,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False)
        self.layer2 = Conv2d(256,
                            512,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            bias=False)
        self.layer3 = Conv2d(512,
                            1024,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            bias=False)
        self.layer4 = Conv2d(1024,
                            2048,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            bias=False)
        self.inner32 = nn.Conv2d(2048, 256, 1, padding=0, stride=1)
        self.inner16 = nn.Conv2d(1024, 256, 1, padding=0, stride=1)
        self.outer16 = nn.Conv2d(256, 256, 3, padding=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        feat4 = self.layer1(x)
        feat8 = self.layer2(feat4)
        feat16 = self.layer3(feat8)
        feat32 = self.layer4(feat16)

        feat32 = self.inner32(feat32)
        feat16 = self.inner16(feat16)
        _, _, H, W = feat16.size()
        feat32 = F.interpolate(feat32, (H, W), mode='nearest')
        feat16 = feat16 + feat32
        return feat4, feat8, feat16, feat32



if __name__ == "__main__":
    resnet = Net()
    dummy_input = torch.randn(1, 3, 224, 224)
    oname = 'resnet50.onnx'
    torch.onnx.export(resnet, dummy_input, oname, verbose=True)

    model = onnx.load(oname)
    img = (cv2.resize(cv2.imread('./000000000139.jpg'), (224, 224)) / 255. - 0.45 ) / 0.225
    img = img.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32)
    rep = backend.prepare(model, device="CUDA:0")
    outputs = rep.run(img)