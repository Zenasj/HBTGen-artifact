# Input: tuple of two tensors (user_ids: (B,), item_ids: (B,)), dtype=torch.int64
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, num_users, num_items):
        super().__init__()
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=8)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=8)
        self.fc1 = nn.Linear(16, 64)
        self.fc1_activation = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc2_activation = nn.ReLU()
        self.output = nn.Linear(32, 1)
        self.out_activation = nn.Sigmoid()

    def forward(self, inputs):
        user_input, item_input = inputs
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)
        vector = torch.cat([user_embedded, item_embedded], dim=-1)
        vector = self.fc1_activation(self.fc1(vector))
        vector = self.fc2_activation(self.fc2(vector))
        pred = self.out_activation(self.output(vector))
        return pred

def my_model_function():
    return MyModel(num_users=13849, num_items=19103)

def GetInput():
    B = 2  # Example batch size; can be any positive integer
    user_input = torch.randint(0, 13849, (B,), dtype=torch.int64)
    item_input = torch.randint(0, 19103, (B,), dtype=torch.int64)
    return (user_input, item_input)

