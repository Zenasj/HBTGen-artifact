import torch
import torch.nn as nn

embag = nn.EmbeddingBag(5, 20)
for X, y in dataloader:
    if torch.is_tensor(X):
        emb = embag(X)
    elif isinstance(X, dict) and '1D_flatten' in X and 'offsets' in X:
        emb = embag(input=X['1D_flatten'], offsets=X['offsets'])

x = {
   'A': {
      'numbers': [[1, 2, 3], [4, 5]]
  }
}