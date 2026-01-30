import torch
from transformers import AutoModelForImageClassification


neural_network = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")
input_data = torch.randn(1, 3, 228, 228)
print(neural_network(input_data))
neural_network_c = torch.compile(neural_network, backend="inductor")
print("Before")
neural_network_c(input_data)
print("After")