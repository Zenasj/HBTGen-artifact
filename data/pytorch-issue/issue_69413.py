3
import torch
import torch.nn as nn

from torch.optim.swa_utils import AveragedModel
from torch.nn.utils import parametrize

if __name__ == '__main__':
    model = nn.Linear(3, 4)
    b = AveragedModel(model)    # This works

    class Param(nn.Module):
        def forward(self, X: torch.Tensor) -> torch.Tensor:
            return X.clamp(0.0, 1.0)

    parametrize.register_parametrization(model, 'weight', Param())

    c = AveragedModel(model)    # This raises the error