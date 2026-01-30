from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
import torch
model = LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf", low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
model = model.eval()
prompt = "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun"

num_iter = 3

with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
    for i in range(num_iter):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        output  = model.generate(input_ids, max_new_tokens=32, do_sample=False, temperature=0.9, num_beams=4)
        gen_text = tokenizer.batch_decode(output, skip_special_tokens=True)
        print(gen_text, flush=True)

from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
import torch
model = LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf", low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
model = model.eval()
prompt = "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun"

import torch._inductor.config as config
config.cpp.enable_kernel_profile=True
config.profiler_mark_wrapper_call=True
torch._dynamo.config.suppress_errors = True
model.generate = torch.compile(model.generate, backend='inductor', dynamic=True)

num_iter = 3
with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
    for i in range(num_iter):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        output  = model.generate(input_ids, max_new_tokens=32, do_sample=False, temperature=0.9, num_beams=4)
        gen_text = tokenizer.batch_decode(output, skip_special_tokens=True)
        print(gen_text, flush=True)

from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
import torch
model = LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf", low_cpu_mem_usage=True)
tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
model = model.eval()
prompt = "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun"
num_iter = 3
for i in range(num_iter):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        output  = model.generate(input_ids, max_new_tokens=32, do_sample=False, temperature=0.9, num_beams=4)
        gen_text = tokenizer.batch_decode(output, skip_special_tokens=True)
        print(gen_text, flush=True)