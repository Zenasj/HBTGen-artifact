import torch
import torch.nn as nn

class M(torch.nn.Module):
    def __init__(self, is_train: bool = False) -> None:
        super().__init__()
        self._is_train = is_train
        self.linear1 = torch.nn.Linear(2, 2)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(2, 2)
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._is_train:
            x = self.relu2(self.linear2(self.relu1(self.linear1(x))))
        return self.linear3(x)

class Model(torch.nn.Module):
    def __init__(self, is_train: bool = False) -> None:
        super().__init__()
        self._m = M(is_train)
        self.linear = torch.nn.Linear(2, 2)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.linear(x))
        x = self._m(x)
        return x

m = Model()

torch.fx.wrap
def some_func(module: torch.nn.Module, x:torch.Tensor) -> torch.Tensor:
    # do something...
    return x

class Model(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.linear(x))
        x = self._m(x)
        x = some_func(self._m, x)
        return x