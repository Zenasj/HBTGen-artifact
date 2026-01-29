# torch.randint(0, 20, (500, 30), dtype=torch.long) ← inferred input shape (batch_size=500, sequence_length=30)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Embedding(500, 10, max_norm=1.0),
            nn.Linear(10, 2)
        )
        self.loss_criterion = nn.NLLLoss()  # Matches original setup despite potential numerical issues

    def forward(self, inputs):
        input, labels = inputs  # Unpack tuple from GetInput()
        em_x = self.layers[0](input).sum(dim=1)
        out = self.layers[1](em_x)
        loss = self.loss_criterion(out, labels)
        return loss

def my_model_function():
    return MyModel()

def GetInput():
    # Generate input and labels as tuple matching the forward's requirements
    input_data = torch.randint(0, 20, (500, 30), dtype=torch.long)
    labels = torch.cat([torch.zeros(250, dtype=torch.long), torch.ones(250, dtype=torch.long)])
    return (input_data, labels)

