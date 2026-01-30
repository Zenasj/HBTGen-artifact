import torch
import torchvision.transforms as transforms

img = torch.rand([1, 1, 256, 256], device="mps");
transform = transforms.RandomPerspective();
img = transform(img);