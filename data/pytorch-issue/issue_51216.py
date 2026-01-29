# torch.rand(B, C, H, W, dtype=torch.float32)  # Input shape (4, 1, 4, 4) as per original example
import torch
import torch.nn as nn
from collections import OrderedDict

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.conv = torch.nn.Conv2d(1, 1, 1)
        self.relu = torch.nn.ReLU()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.conv(x)
        x = self.relu(x)
        x = self.dequant(x)
        return x

    # Fix for pickling issue: initialize missing hooks
    def __setstate__(self, state):
        super().__setstate__(state)
        mods = list(self._modules.values())
        while mods:
            mod = mods.pop(0)
            for attr in [
                '_forward_pre_hooks',
                '_forward_hooks',
                '_backward_hooks',
                '_state_dict_hooks',
                '_load_state_dict_pre_hooks',
                '_non_persistent_buffers_set',
                '_modules'
            ]:
                if not hasattr(mod, attr):
                    setattr(mod, attr, 
                        OrderedDict() if attr.endswith('hooks') else set() if attr.endswith('set') else OrderedDict()
                    )
            mods += list(getattr(mod, '_modules', {}).values())

def my_model_function():
    model = MyModel()
    model.eval()  # Required for quantization preparation
    return model

def GetInput():
    return torch.rand(4, 1, 4, 4, dtype=torch.float32)

