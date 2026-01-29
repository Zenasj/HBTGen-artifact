# torch.rand(B, 3, 224, 224, dtype=torch.float32)  # Example input shape for image-like data
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = nn.Linear(224*224*3, 100)
        self.layer2 = nn.Linear(100, 10)

    def forward(self, x):
        with torch.autocast(device_type="cuda", enabled=True):  # Example usage of autocast
            x = x.view(x.size(0), -1)
            x = self.layer1(x)
            x = self.layer2(x)
        return x

def my_model_function():
    model = MyModel()
    # Initialize weights (placeholder initialization)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
    return model

def GetInput():
    batch_size = 4  # Example batch size
    return torch.rand(batch_size, 3, 224, 224, dtype=torch.float32)

