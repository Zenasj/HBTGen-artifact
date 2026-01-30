import torch.nn as nn

model = nn.Transformer()
sz = 3
original_mask = model.generate_square_subsequent_mask(sz = sz)
print(original_mask)

tensor([[0., 0., 0.],
        [-inf, 0., 0.],
        [-inf, -inf, 0.]])

tensor([[0., -inf, -inf],
        [0., 0., -inf],
        [0., 0., 0.]])