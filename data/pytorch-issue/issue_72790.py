# torch.rand(1, 3, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn
from torchvision import models

def _sort(x, dim=0, keepdim=False):
    x_sum = torch.sum(x, dim, keepdim)
    x_rand = torch.randn(x_sum.shape, device=x.device)
    return torch.maximum(x_sum, x_rand)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = _sort(x, dim=1, keepdim=True)
        max_out, _ = torch.min(x, dim=1, keepdim=True)  # As in original code
        scale = torch.cat([avg_out, max_out], dim=1)
        scale = self.conv(scale)
        out = torch.maximum(x, self.sigmoid(scale))
        return out

class MyModel(nn.Module):  # Renamed from SpatialFeatureExtractor
    def __init__(self, onnx_batch_size):
        super(MyModel, self).__init__()
        self.onnx_batch_size = onnx_batch_size
        vgg11_model = models.vgg11_bn(pretrained=True)
        vgg11_part1 = vgg11_model.features[:11]
        vgg11_part2 = vgg11_model.features[11:15]
        for param in vgg11_part1.parameters():
            param.requires_grad = False
        for param in vgg11_part2.parameters():
            param.requires_grad = True

        self.feature_extractor_p1 = nn.Sequential(vgg11_part1)
        self.spatialAtt = SpatialAttention()
        self.feature_extractor_p2 = nn.Sequential(
            vgg11_part2,
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        p1_features = self.feature_extractor_p1(x)
        features_after_attention = self.spatialAtt(p1_features)
        p2_features = self.feature_extractor_p2(features_after_attention)
        return p2_features.view(self.onnx_batch_size, -1, 1, 1)

def my_model_function():
    return MyModel(1)  # Matches GetInput() batch size

def GetInput():
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

