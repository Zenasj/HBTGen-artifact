import torch
import torchvision.transforms as T

from PIL import Image

transforms = T.Compose([
    T.Resize(size=224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img = Image.open(path_to_image).convert('RGB')
data = transforms(img)