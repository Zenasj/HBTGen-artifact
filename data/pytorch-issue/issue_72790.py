import torch
import torch.nn as nn
from torchvision import models

def _sort(x, dim=0, keepdim=False):
    x_sum = torch.sum(x, dim, keepdim)
    x_rand = torch.randn(x_sum.shape).cuda()
    return torch.maximum(x_sum, x_rand)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = _sort(x, dim=1, keepdim=True)
        max_out, _ = torch.min(x, dim=1, keepdim=True)
        scale = torch.cat([avg_out, max_out], dim=1)
        scale = self.conv(scale)
        out = torch.maximum(x, self.sigmoid(scale))
        return out


class SpatialFeatureExtractor(nn.Module):
    def __init__(self, onnx_batch_size):
        super(SpatialFeatureExtractor, self).__init__()
        self.onnx_batch_size = onnx_batch_size
        vgg11_model = models.vgg11_bn(pretrained=True)
        vgg11_part1 = vgg11_model.features[:11]
        vgg11_part2 = vgg11_model.features[11:15]
        for param in vgg11_part1.parameters():
            param.requires_grad = False
        for param in vgg11_part2.parameters():
            param.requires_grad = True

        self.feature_extractor_p1 = nn.Sequential(
            vgg11_part1
        )

        self.spatialAtt = SpatialAttention()

        self.feature_extractor_p2 = nn.Sequential(
            vgg11_part2,
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        p1_features = self.feature_extractor_p1(x)
        features_after_attention = self.spatialAtt(p1_features)

        p2_features = self.feature_extractor_p2(features_after_attention)
        features = p2_features.view(self.onnx_batch_size, -1, 1, 1)
        
        return features

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image = torch.zeros(1, 3, 224, 224).to(device)

    network = SpatialFeatureExtractor(1).to(device)
    network.train(False)

    torch.onnx.export(network, (image), "feature_extractor.onnx", verbose=False, \
                      input_names=['input'],
                      output_names=['features'])