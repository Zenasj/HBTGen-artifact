python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "gpt2" # any LM would result the same
tokenizer = AutoTokenizer.from_pretrained(path) 
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map={"":"mps"})

t = tokenizer("anything", return_attention_mask=False, return_tensors='pt')
with torch.inference_mode():
    model(**t)