# torch.rand(16, 3, 224, 224, dtype=torch.float32).cuda()  # Input shape for MyModel
import torch
import torchvision.models as models

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = models.resnet18()

    def forward(self, x):
        return self.model(x)

def my_model_function():
    model = MyModel()
    model.cuda()  # Move model to CUDA device
    return model

def GetInput():
    return torch.rand(16, 3, 224, 224, dtype=torch.float32).cuda()  # Random input matching ResNet18 requirements

