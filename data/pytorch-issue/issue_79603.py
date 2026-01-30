import pytorch_geometric.nn as pyg_nn

model = pyg_nn.GIN(512, 512, num_layers=5, jk="LSTM")

import torch.nn as nn
from torch_geometric.nn import global_add_pool

class DeepMultisets(nn.Module):
    def __init__(
        self, in_channels: int, num_outputs: int, hidden_channels: int, **kwargs
    ):
        super().__init__()
        self.fc_node = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
        )

        self.fc_global_1 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
        )

        self.fc_global_2 = nn.Linear(hidden_channels, num_outputs)

    def forward(self, data):
        x, batch = data.x, data.batch
        x = data.x.float()
        x = self.fc_node(x)
        x = global_add_pool(x, batch)
        x = self.fc_global_1(x)
        x = self.fc_global_2(x)
        return x