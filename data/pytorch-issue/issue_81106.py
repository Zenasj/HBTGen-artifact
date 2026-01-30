import torch
import torch.nn as nn
import torchvision

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, kernel_size=5, stride=1, bias=False)

    def forward(self, x):
        print('before', x.device, x.dtype, x.layout, x.is_contiguous())
        # x = torchvision.transforms.functional.rgb_to_grayscale(x, num_output_channels=1)  # FAIL
        # x = x.sum(dim=-3).unsqueeze(dim=-3)  # FAIL
        # x = x[:, 0, ...].unsqueeze(dim=-3)  # FAIL
        x = x[:, :1, ...]  # OK
        
        print('after', x.device, x.dtype, x.layout, x.is_contiguous())
        x = self.conv1(x)
        return x