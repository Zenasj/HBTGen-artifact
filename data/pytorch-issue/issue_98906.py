# torch.rand(B, 3, 224, 224, dtype=torch.float32)
import torch
import torchvision.models as models
import copy

def make_contiguous(module):
    with torch.no_grad():
        state_dict = module.state_dict()
        state_dict = copy.deepcopy(state_dict)
        for name, param in state_dict.items():
            if not param.is_contiguous():
                state_dict[name] = param.contiguous()
        module.load_state_dict(state_dict, assign=True)

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = models.resnet50()
        self.model = self.model.to(memory_format=torch.channels_last)
        make_contiguous(self.model)
        
    def forward(self, x):
        return self.model(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

