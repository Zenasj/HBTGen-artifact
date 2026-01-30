import torch.nn as nn
import torch


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        input_size = 10
        hidden_size = 20

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)

    def forward(self, input):
        hidden_states = self.embedding(input)
        self.lstm(hidden_states)


model = Model()
model = model.to("cuda").to(torch.bfloat16)

input_tensor = torch.tensor(range(5), dtype=torch.long, device="cuda")
model(input_tensor)