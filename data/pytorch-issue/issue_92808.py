# pip install transformers
from transformers import BertModel
import torch

model = BertModel.from_pretrained("bert-base-uncased")