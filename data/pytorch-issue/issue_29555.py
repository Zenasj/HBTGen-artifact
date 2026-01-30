import torch.nn as nn

class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.embedding_to_learn = nn.Embedding(10, 5, requires_grad=True)
    self.embedding_to_not_learn = nn.Embedding(10, 5, requires_grad=False)

class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.embedding_to_learn = nn.Embedding(10, 5)
    self.embedding_to_not_learn = nn.Embedding(10, 5)

model = Net()
model.embedding_to_not_learn.requires_grad_(requires_grad=False)