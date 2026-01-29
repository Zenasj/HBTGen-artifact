# torch.rand(B, 1000, dtype=torch.float32)
import torch
import torch.nn as nn

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

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model1 = SpectralNormTestModel()
        self.model2 = SpectralNormTestModel()
        
        # Pre-initialize model1 via a train forward to ensure spectral norm is initialized
        dummy_input = torch.randn(1, 1000, requires_grad=False)
        self.model1.train()
        with torch.no_grad():
            _ = self.model1(dummy_input)
        self.model1.eval()  # restore to eval mode

    def forward(self, x):
        out1 = self.model1(x)  # pre-initialized
        out2 = self.model2(x)  # uninitialized (first eval forward)
        # Return whether outputs differ beyond a threshold (problem occurs when True)
        return torch.abs(out1 - out2).mean() > 1e-3

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 1000, dtype=torch.float32)

