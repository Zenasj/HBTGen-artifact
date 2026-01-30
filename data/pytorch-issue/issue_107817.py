import torch
import torch.nn as nn

# Create some example data
input = torch.tensor(
    [
        [0.8, 0.2, -0.5],
        [0.1, 0.9, 0.3],
    ]
)
target1 = torch.tensor(
    [
        [1, 0, 1],
        [0, 1, 1],
        [0, 1, 1],
    ]
)
target2 = torch.tensor(
    [
        [1, 0],
        [0, 1],
    ]
)
target3 = torch.tensor(
    [
        [1, 0, 1],
        [0, 1, 1],
    ]
)
loss_func = nn.MultiLabelSoftMarginLoss()
try:
    loss = loss_func(input, target1).item()
except RuntimeError as e:
    print('target1 ', e)
try:
    loss = loss_func(input, target2).item()
except RuntimeError as e:
    print('target2 ', e)
loss = loss_func(input, target3).item()
print('target3 ', loss)