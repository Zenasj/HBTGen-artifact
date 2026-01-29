# torch.randint(0, 2, (), dtype=torch.long)  # Inferred input shape

import torch
from torch import nn

class TestModule3(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeds = nn.Embedding(2, 7)

class TestModule2(nn.Module):
    def __init__(self, embed3):
        super().__init__()
        self.embeds = nn.Embedding(2, 6)
        self.embed3 = embed3  # Reference to shared TestModule3 instance

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Create shared TestModule3 instance
        shared_model3 = TestModule3()
        # Create TestModule2 with shared_model3
        model2 = TestModule2(shared_model3)
        # Main embedding layers
        self.embeds = nn.Embedding(2, 5)
        self.embeds2 = model2  # EmbedModule2 instance
        self.embeds3 = shared_model3  # Direct reference to shared TestModule3

    def forward(self, id1):
        return self.embeds(id1)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(0, 2, (), dtype=torch.long)

