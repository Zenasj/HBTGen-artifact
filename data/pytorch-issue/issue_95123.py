import torch.nn as nn

padding = _pair(padding)

class MyModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv2dt = nn.ConvTranspose2d(
      in_channels=16,
      out_channels=32,
      kernel_size=(3, 3),
      stride=2,
      padding="same"
    )
  
  def forward(self, x):
    print(self.conv2dt.padding) # ('s', 'a', 'm', 'e')
    y = self.conv2dt(x) # Error triggered in this line, have a look below