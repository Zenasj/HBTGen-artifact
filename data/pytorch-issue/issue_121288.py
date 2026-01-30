import torch
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased").eval()

text = "bert base inference with torch.compile()"
encoded_input = tokenizer(text, return_tensors='pt')

model.eval()
model = torch.compile(model)

with torch.set_grad_enabled(False):
    out = model(**encoded_input)
    print(out)