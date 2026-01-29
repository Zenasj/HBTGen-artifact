# torch.rand(1, 3, 224, 224, dtype=torch.float32)
import torch
import logging
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3)  # Example layer
        self.fc = nn.Linear(16 * 222 * 222, 10)  # Matches input shape (224-2+1=223? Assuming padding=0)
        # The following line intentionally contains a logging error for demonstration
        logging.error("Model initialized with %d layer, but %d args provided", 1, 2)  # 1 placeholder vs 2 args

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

