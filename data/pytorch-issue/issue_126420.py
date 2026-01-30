import torch.nn as nn
import torchvision

from torchvision.transforms import v2 as T
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch
from torch.sparse import to_sparse_semi_structured
from torch import nn
from pathlib import Path
from torch.optim import SGD

test_dataloader = DataLoader(
    MNIST(
        root=Path('data'),
        train=False,
        download=True,
        transform=T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=(0.1307,), std=(0.3081,)),
            T.ToDtype(torch.float16)
        ])
    ),
    batch_size=8
)

class SparseLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        density: float = 0.1,
    ):
        super(SparseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features


        self.weight = nn.Parameter(to_sparse_semi_structured((torch.rand(self.out_features, self.in_features) < density).half().cuda()))
        self.bias = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias)


model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=28*28, out_features=512, dtype=torch.float16),
        nn.ReLU(),
        SparseLinear(512, 256, density=0.1, bias=False),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=10, dtype=torch.float16),
        nn.Softmax(dim=1)
)

device = torch.device(0)
model = model.to(device)


loss_function = nn.CrossEntropyLoss()
optimizer = SGD(params=model.parameters(), lr=1e-4)

x, y_true = next(iter(test_dataloader))

optimizer.zero_grad()
y_pred = model(x.to(device))
loss = loss_function(y_pred, F.one_hot(y_true, num_classes=10).half().to(device))
loss.backward()