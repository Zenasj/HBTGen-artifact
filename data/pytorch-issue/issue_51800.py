import torch.nn as nn

from torch import nn
import torch

class SpectralNormTestModel(nn.Module):
        def __init__(self):
            super(SpectralNormTestModel, self).__init__()
            feature_count = 1000

            def init_(layer: nn.Linear):
                nn.init.orthogonal_(layer.weight.data, 1.372)
                nn.init.zeros_(layer.bias.data)

            self.fc1 = nn.Linear(feature_count, feature_count)
            init_(self.fc1)
            self.fc1 = nn.utils.spectral_norm(self.fc1)
            self.act1 = nn.PReLU(feature_count)

            self.fc2 = nn.Linear(feature_count, feature_count)
            init_(self.fc2)
            self.fc2 = nn.utils.spectral_norm(self.fc2)
            self.act2 = nn.PReLU(feature_count)

            self.out = nn.Linear(feature_count, 1)
            init_(self.out)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.out(self.act2(self.fc2(self.act1(self.fc1(x)))))
     
def spectral_norm_init_test(use_forward: bool):
    x = torch.randn(10000, 1000, device='cuda')
    model = SpectralNormTestModel().cuda()
    if use_forward:
        y = model.forward(x)
    y = model.eval()(x) 
    print(f"mean absolute output: {y.abs().mean()}")

print('----- test with use_forward ------')
spectral_norm_init_test(True)
print('----- test without use_forward ------')
spectral_norm_init_test(False)
print('----- done ------')