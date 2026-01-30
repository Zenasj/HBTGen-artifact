import torch.nn as nn
print(nn.modules.transformer.Transformer.generate_square_subsequent_mask(4, device='mps'))

tensor([[nan, -inf, -inf, -inf],
        [nan, nan, -inf, -inf],
        [nan, nan, nan, -inf],
        [nan, nan, nan, nan]], device='mps:0')

tensor([[0., -inf, -inf, -inf],
        [0., 0., -inf, -inf],
        [0., 0., 0., -inf],
        [0., 0., 0., 0.]])

[tasklist]
### Tasks