from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
from itertools import chain

# load model
print("Loading model...")
model_id = "decapoda-research/llama-7b-hf"
model = LlamaForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float32
)
tokenizer = LlamaTokenizer.from_pretrained(model_id)
print("Model loaded")
model = model.eval()

import torch._inductor.config as config
torch._dynamo.config.assume_static_by_default = False
model.generate = torch.compile(model.generate, dynamic=True)

print("Model initialized")

# input prompt
prompt = "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun"

input_size = tokenizer(prompt, return_tensors="pt").input_ids.size(dim=1)
print("---- Prompt size:", input_size)

generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=4)
with torch.inference_mode(): # no problem without this
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output = model.generate(
        input_ids, max_new_tokens=32, **generate_kwargs
    )
    gen_ids = output
    gen_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)