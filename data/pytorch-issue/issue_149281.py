import torch
import torch.nn as nn

sequence_size = 32
env_size = 64
input_dim = 39
hidden_dim = 64
output_dim = 6
device = "cuda"

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

batch_input = torch.randn((sequence_size, env_size, input_dim), dtype=torch.float32, device=device)

model = nn.Linear(in_features=input_dim, out_features=output_dim, device=device)
batch_output = model(batch_input)
print("big batch together:", batch_output[0,0])
print("smaller batch:", model(batch_input[0])[0])