python
import torch

import onnx
from transformers import BertTokenizer, BertModel

device = torch.device("cuda:0")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model = model.to(device)

text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt').to(device="cuda:0")


opt_model = torch.compile(model, mode='max-autotune', fullgraph=True)
torch.onnx.export(opt_model, tuple(encoded_input.values()), 
						f='bert_triton.onnx',  
						input_names=['input_ids'], 
						output_names=['logits'], 
						dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence'}, 
						  			'logits': {0: 'batch_size', 1: 'sequence'}}, 
						do_constant_folding=True, 
						opset_version=13, 
				)