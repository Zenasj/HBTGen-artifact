# torch.rand(B, 3, 224, 224, dtype=torch.float32)
import torch
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models._utils import IntermediateLayerGetter
from torch import nn

class FrozenBatchNorm2d(torch.nn.Module):
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = models.resnet50(pretrained=True, norm_layer=FrozenBatchNorm2d)
        self.model1 = create_feature_extractor(self.base_model, return_nodes={'layer4': '0'})
        self.model2 = IntermediateLayerGetter(self.base_model, return_layers={'layer4': '0'})

    def forward(self, x):
        out1 = self.model1(x)
        out2 = self.model2(x)
        # Return boolean as tensor to comply with model output requirements
        return torch.tensor(
            torch.allclose(out1['0'], out2['0'], atol=1e-5, rtol=1e-5),
            dtype=torch.bool
        )

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

