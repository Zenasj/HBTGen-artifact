from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
import torch
model = LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf", low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
model = model.eval()
prompt = "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun"
input_size = tokenizer(prompt, return_tensors="pt").input_ids.size(dim=1)

num_iter = 3
num_warmup = 2
def trace_handler(prof):
    print(prof.key_averages().table(
        sort_by="self_cpu_time_total", row_limit=-1))
with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU],
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=1),
            on_trace_ready=trace_handler
            ) as prof:
  with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
    for i in range(num_iter):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        output  = model.generate(input_ids, max_new_tokens=32, do_sample=False, temperature=0.9, num_beams=4)
        gen_text = tokenizer.batch_decode(output, skip_special_tokens=True)
        print(gen_text, flush=True)
        prof.step()