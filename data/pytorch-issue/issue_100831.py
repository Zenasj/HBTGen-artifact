import torch.nn as nn

py
import torch

torch.manual_seed(420)

x = torch.randint(0, 100, (1, 10))

class Model(torch.nn.Module):

    def __init__(self, input_size, num_embeddings):
        super(Model, self).__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=input_size)

    def forward(self, input_ids):
        embeddings = self.embedding(input_ids)
        return embeddings

input_size = 1
num_embeddings = 1
func = Model(input_size, num_embeddings).to('cpu')

with torch.no_grad():
    func.train(False)

    jit_func = torch.compile(func)
    res2 = jit_func(x)
    print(res2)

    res1 = func(x) # without jit
    # IndexError: index out of range in self