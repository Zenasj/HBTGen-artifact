from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch
from torch._inductor import config
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To prevent long warnings :)
 
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.bfloat16)
 
torch._inductor.config.cpp_wrapper = True
model.forward = torch.compile(model.forward, dynamic=True)
 
batch_size = 2
input_text = ["The theory of special relativity states "] * batch_size
print(input_text)
print(f"batch size is {len(input_text)}")
tokenizer.pad_token = tokenizer.eos_token
input_ids = tokenizer(input_text, return_tensors="pt", padding=True)
input_shape = input_ids["input_ids"].shape
print(F"input ids shape is {input_shape}")
 
generation_kwargs = {"do_sample": False, "num_beams": 2, "max_new_tokens": 32, "min_new_tokens": 32}
 
outputs = model.generate(**input_ids, **generation_kwargs)
 
outputs = model.generate(**input_ids, **generation_kwargs)
outputs = model.generate(**input_ids, **generation_kwargs)
outputs = model.generate(**input_ids, **generation_kwargs)
outputs = model.generate(**input_ids, **generation_kwargs)
outputs = model.generate(**input_ids, **generation_kwargs)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))