import torch
from torch._export import capture_pre_autograd_graph
from transformers import BertTokenizer, BertModel, AutoConfig

model_id = 'bert-base-uncased'
config = AutoConfig.from_pretrained(model_id, return_dict=False)
tokenizer = BertTokenizer.from_pretrained(model_id, return_dict=False)
model = BertModel.from_pretrained(model_id, config=config)

text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')

with torch.no_grad():
    captured_model = capture_pre_autograd_graph(model, (), kwargs=encoded_input)
    print('Success')