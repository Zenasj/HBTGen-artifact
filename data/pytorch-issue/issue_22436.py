# torch.rand(1, 3, dtype=torch.float32) ← Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.generator = nn.Sequential(
            nn.Linear(3, 2),
            nn.ReLU(),
            nn.Linear(2, 2)
        )
        self.discriminator = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Generator forward pass
        gen_output = self.generator(x)
        
        # Discriminator forward pass
        disc_output = self.discriminator(gen_output)
        
        return gen_output, disc_output

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(1, 3, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')

