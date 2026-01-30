import torch
import torch.nn as nn

def test_embedding(self):
        def f(a, b):
            return torch.nn.functional.embedding(a, b)

        input = torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]])
        embedding_matrix = torch.rand(10, 3)
        r = make_fx(f, tracing_mode="symbolic")(input, embedding_matrix)
        print(r)