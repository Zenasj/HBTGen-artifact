import torch.nn as nn

import os
import torch

import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms


model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
model.eval()

batchsize = 50000
workers = 1
dataset_path = 'data/imagenet_2012/'
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
val_data = datasets.ImageFolder(root=os.path.join(dataset_path, 'val'), transform=transforms.Compose( [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize,]))
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batchsize, num_workers=workers)


batch = next(iter(val_loader))
images, labels = batch

with torch.no_grad():
    output = model(images)


for i in output:
    out_soft = torch.nn.functional.softmax(i, dim=0)
    print(int(torch.argmax(out_soft)))