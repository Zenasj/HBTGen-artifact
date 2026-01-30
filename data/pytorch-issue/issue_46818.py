import torch.nn as nn

def forward(self, X):
        X = nn.Linear(self.num_features, self.num_hiddens)(X)
        X = nn.BatchNorm1d(self.num_hiddens).train(self.training)(X)
        ...