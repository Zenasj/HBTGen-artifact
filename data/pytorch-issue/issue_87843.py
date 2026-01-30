import torch.nn as nn

import torch
from transformers import AutoConfig, AutoModel
model = AutoModel.from_pretrained("bert-base-uncased")
model_quantized = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
model_quantized.save_pretrained("quantized/")

dtype

save_pretrained