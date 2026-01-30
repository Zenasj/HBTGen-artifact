import torchvision

import torch
import random

# This problem only occurs on the CPU, not the GPU.
device = torch.device("cpu")

# Uncomment the following line and the problem goes away.
# device = torch.device("cuda:0")

from torchvision.models.resnet import resnext50_32x4d
model = resnext50_32x4d(pretrained=False).to(device)
_ = model.eval()

def predict():
    # Using this line, memory usage blows up (but only on the CPU!)
    x = torch.zeros((random.randint(10, 30), 3, 224, 224)).float()

    # Using the following line, memory usage is constant.
    #x = torch.zeros((30, 3, 224, 224)).float()

    print("batch size is:", x.shape[0])

    with torch.no_grad():
        y_pred = model(x)
        return y_pred.mean().item()

predictions = []
for i in range(100):
    y_pred = predict()
    predictions.append(y_pred)