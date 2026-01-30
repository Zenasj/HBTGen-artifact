import torch
import torch.nn as nn
import torch.nn.functional as F
import ai_edge_torch

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

model = SimpleNN()

example_input = torch.randn(1, 1, 28, 28)

edge_model = ai_edge_torch.convert(model.eval(), (example_input,))

edge_model.export('simple_nn.tflite')