from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler
import torch

with torch.device("cuda"):
    model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

inp = tokenizer("Today I am in Paris and", return_tensors="pt").to("cuda")

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    on_trace_ready=tensorboard_trace_handler(dir_name=f"./tb_logs/mylog"),
):
    res = model.generate(**inp, num_beams=1, max_new_tokens=1)