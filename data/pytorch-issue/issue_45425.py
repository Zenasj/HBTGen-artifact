import torch.nn as nn
import torchvision

class DecoderBlock(nn.Module, ABC):
    def __init__(self, in_depth, middle_depth, out_depth):
        super(DecoderBlock, self).__init__()
        self.conv_relu = ConvRelu(in_depth, middle_depth)
        self.conv_transpose = nn.ConvTranspose2d(middle_depth, out_depth, kernel_size=4, stride=2, padding=1)
        self.activation = nn.ReLU(inplace=True)

class UNetResNet(nn.Module):
    def __init__(self, num_classes):
        super(UNetResNet, self).__init__()
        self.encoder = torchvision.models.resnet101(pretrained=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu, self.pool)
        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv4 = self.encoder.layer4