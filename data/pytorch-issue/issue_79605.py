import torch.nn as nn

block = nn.Sequential(
            nn.Linear(n_input, n_feat),
            LayerNorm2DSlim(n_feat, relu=True),
            nn.Linear(n_feat, n_feat),
        )