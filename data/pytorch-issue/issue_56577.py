# torch.rand(2, 3, 3, dtype=torch.long)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # num_embeddings set to 20 to handle indices up to 19 (original input had max 18 after fixing negatives)
        self.embedding = nn.Embedding(num_embeddings=20, embedding_dim=21)
        self.cuda()  # Matches original code's .cuda() call

    def forward(self, x):
        # Convert to IntTensor as in original code (indices must be int/long)
        return self.embedding(x.type(torch.cuda.IntTensor))

def my_model_function():
    return MyModel()

def GetInput():
    # Generates valid indices (0-19) for num_embeddings=20
    return torch.randint(0, 20, (2, 3, 3), dtype=torch.long).cuda()

