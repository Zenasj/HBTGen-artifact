import torch
import torch.nn as nn
import torch.nn.functional as F
import json

checkpoint = torch.jit.load('./model.pt')
for name, param in checkpoint.named_parameters():
    print(f"Layer: {name}")
    print(f"Weight:\n{param.data}")
    print(f"Bias:\n{param.data}")
    print("\n")


input_file = "./torch_origin_model.input.0.json"

with open(input_file, 'r') as file:
    data = json.load(file)


input_list = []
for key, value in data.items():
    input_list.append(value)


tensor_data = torch.tensor(input_list[0], dtype=torch.float32)

checkpoint.eval()

with torch.no_grad():
    output = checkpoint(tensor_data)

print('---------')

print(output)