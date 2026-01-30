python
import io
import torch
with open('resnet18-5c106cde.pth', 'rb') as f:
    buffer = io.BytesIO(f.read())
torch.load(buffer)