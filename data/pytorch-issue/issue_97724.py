import torch

from transformers import DistilBertTokenizer, DistilBertModel
text = "Replace me by any text you'd like."
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
encoded_input = tokenizer(text, return_tensors="pt")

from torch._dynamo.utils import clone_inputs
clone_inputs(encoded_input) # this won't preserve the dict structure of the original inputs