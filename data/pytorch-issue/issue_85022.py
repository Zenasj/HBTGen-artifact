import torchvision

from torchvision.models.vgg import vgg16_bn
model = vgg16_bn(pretrained=True)