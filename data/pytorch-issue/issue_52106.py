import torch.nn as nn

3
from typing import List, Dict
import torch
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, labels: torch.Tensor):
        self.compute_loss(labels)
    def compute_loss(self, labels: Dict[str, object]):
        return None

torch.jit.script(MyModel())

3
from typing import List, Dict
import torch
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, labels: Dict[str, object]):
        return labels

torch.jit.script(MyModel())