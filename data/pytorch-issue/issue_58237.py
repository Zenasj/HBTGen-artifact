# torch.rand(1, 3, 224, 224, dtype=torch.float32)
import torch
import torchvision.transforms.functional as F
import torch.hub

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = torch.hub.load("pytorch/vision:v0.9.0", "mobilenet_v2", pretrained=True)
        self.model.eval()  # Matches original model setup before tracing
    
    def forward(self, x):
        return self.model(x)

def my_model_function():
    # Returns an instance of MyModel with pretrained weights and eval mode
    return MyModel()

def GetInput():
    # Returns a normalized random input tensor matching MobileNetV2 requirements
    return F.normalize(
        torch.rand(1, 3, 224, 224, dtype=torch.float32),
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

