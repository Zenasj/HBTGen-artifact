import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.amp import autocast
model_name = "/models/Llama-2-7b-chat-hf/"
#model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = "cuda"
amp_dtype = torch.float16
block = model.model.layers[0]
block = block.to(device).to(amp_dtype)

input_ids = torch.randn(8, 2048, 4096).to(amp_dtype).to(device)
input_others = {
    "attention_mask": torch.randn(8, 1, 2048, 2048).to(amp_dtype).to(device),
    "position_ids": torch.arange(2048).unsqueeze(0).to(torch.int64).to(device),
    "cache_position": torch.arange(2048).to(torch.int64).to(device),
}

opt_block = torch.compile(block)
#opt_block     = block
out_without_amp = opt_block.forward(input_ids, **input_others)
print(f"out_without_amp[0].shape: {out_without_amp[0].shape}")

with autocast(device_type=device, dtype=amp_dtype):
    out = opt_block.forward(input_ids, **input_others)
    print(f"out[0].shape: {out[0].shape}")