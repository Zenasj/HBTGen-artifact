import torch.nn as nn

import torch
from transformers import AutoConfig,AutoModel
model = AutoModel.from_pretrained("bert-base-uncased")
model_quant = torch.quantization.quantize_dynamic(model,{torch.nn.Linear}, dtype=torch.qint8)
torch.jit.save(model_quant, 'quantized.pt')

torch.jit.save(model, 'quantized.pt')

state_dict

import torch
from transformers import AutoConfig,AutoModel
model = AutoModel.from_pretrained("bert-base-uncased")
model_quant = torch.quantization.quantize_dynamic(model,{torch.nn.Linear}, dtype=torch.qint8)
quantized_state_dict = model_quant.state_dict()
torch.jit.save(quantized_state_dict, 'scriptmodule.pt')

import torch
from transformers import BertConfig, BertModel
model = BertModel.from_pretrained("bert-base-uncased")
model_quant = torch.quantization.quantize_dynamic(model,{torch.nn.Linear}, dtype=torch.qint8)
torch.jit.save(model_quant, 'quantized.pt')

torch.trace