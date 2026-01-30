import torch.nn as nn

from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights
img = read_image("dog2.jpg").to("mps")
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights).to("mps")
model.eval()
preprocess = weights.transforms()
batch = preprocess(img).unsqueeze(0)
prediction = model(batch).squeeze(0).softmax(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()

import torch
import torchvision
import matplotlib.pyplot as plt
img=torchvision.io.read_image("assets/dog2.jpg").to("mps")
y=torch.nn.functional.interpolate(img.unsqueeze(0).to(torch.float32), size=(232, 232), mode='bilinear').squeeze(0).to(torch.uint8)
plt.imshow(y.cpu().permute(1, 2, 0))
plt.show()