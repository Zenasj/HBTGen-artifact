# torch.tensor([2], dtype=torch.long)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Problematic setup (original issue's model)
        self.emb1 = nn.Embedding(10, 5)
        self.lin1 = nn.Linear(5, 1)
        self.seq1 = nn.Sequential(self.emb1, self.lin1)  # Problematic: uses existing attributes
        
        # Fixed setup (workaround)
        self.seq2 = nn.Sequential(
            nn.Embedding(10, 5),
            nn.Linear(5, 1)
        )  # Fixed: layers defined inline
        
        # Second test case (problematic with Sigmoid)
        self.emb2 = nn.Embedding(8, 10)
        self.seq3 = nn.Sequential(
            self.emb2,
            nn.Linear(10, 1),
            nn.Sigmoid()
        )  # Problematic: uses existing emb2

    def forward(self, x):
        # Return outputs from all setups for comparison
        out1 = self.seq1(x)   # Problematic
        out2 = self.seq2(x)   # Fixed
        out3 = self.seq3(x)   # Problematic (second test case)
        return out1, out2, out3

def my_model_function():
    return MyModel()

def GetInput():
    return torch.tensor([2], dtype=torch.long)

