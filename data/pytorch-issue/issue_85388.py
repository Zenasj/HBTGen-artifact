# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Simplified SegNet structure with max_unpool2d operator
        self.enc_conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.dec_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dec_conv2 = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x):
        # Encoder path with maxpool indices tracking
        x = self.enc_conv1(x)
        size1 = x.size()  # Save input size before pooling
        x, indices1 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        
        x = self.enc_conv2(x)
        size2 = x.size()
        x, indices2 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        
        # Decoder path using stored indices and output sizes
        x = F.max_unpool2d(x, indices2, kernel_size=2, stride=2, output_size=size2)
        x = self.dec_conv1(x)
        
        x = F.max_unpool2d(x, indices1, kernel_size=2, stride=2, output_size=size1)
        x = self.dec_conv2(x)
        
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 3, 224, 224, dtype=torch.float32)

