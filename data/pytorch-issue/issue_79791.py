# torch.rand(1, 50)  # Assuming the input is a batch of tokenized sentences with a fixed length of 50
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define a simple embedding layer for demonstration
        self.embedding = nn.Embedding(10000, 128)  # Vocabulary size of 10000, embedding dimension of 128
        self.fc = nn.Linear(128 * 50, 128)  # Fully connected layer to produce the final embedding

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # Assuming the input is a batch of tokenized sentences with a fixed length of 50
    return torch.randint(0, 10000, (1, 50))  # Random integers between 0 and 9999

