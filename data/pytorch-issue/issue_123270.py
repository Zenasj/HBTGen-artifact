import torch
import torch.nn as nn
import torch.nn.functional as F

def dynamic_quantization(model):
    model_quantized = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    return model_quantized

model_dynamic_quantization = dynamic_quantization(model)

device = 'cpu'
model_dynamic_quantization.eval()  # Set the model to evaluation mode
model.to(device)
with torch.no_grad():  # No need to track gradients during evaluation
    for batch in test_dataloader:
                data, labels = batch
                data, labels = audio_data.to(device), labels.to(device)
                outputs = model(data)