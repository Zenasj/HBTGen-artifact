from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
from transformers.cache_utils import StaticCache

tokenizer = AutoTokenizer.from_pretrained(
    "NousResearch/Llama-2-7b-chat-hf", padding_side="left", pad_token="<s>"
)

with torch.device("cuda"):
    model = AutoModelForCausalLM.from_pretrained(
        "NousResearch/Llama-2-7b-chat-hf",
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )

inputs = tokenizer(
    ["I would", "Today I am in Paris and"], padding=True, return_tensors="pt"
).to(model.device)

new_tokens = 10
gen_config = GenerationConfig(
    max_new_tokens=new_tokens,
    min_new_tokens=new_tokens,
    use_cache=True,
    pad_token_id=tokenizer.pad_token_id,
    num_beams=1,
    do_sample=False,
    eos_token_id=None,  # This is required for min_new_tokens to actually have an effect.
)
model.generation_config.eos_token_id = None  # greedy_search falls back on this eos_token_id that we need to set to None as well for min_new_tokens to have an effect.

gen_out = model.generate(**inputs, generation_config=gen_config)

decoded = tokenizer.batch_decode(gen_out, skip_special_tokens=True)

print("decoded", decoded)

print("compiling...")

model.forward = torch.compile(model.forward, mode="reduce-overhead")
print("Finished compile call")

# warmup
gen_out = model.generate(**inputs, generation_config=gen_config, cache_implementation="static")

print("\n\n\n\n\n\n----- second call")
gen_out = model.generate(**inputs, generation_config=gen_config, cache_implementation="static")

decoded = tokenizer.batch_decode(gen_out, skip_special_tokens=True)

print("decoded static", decoded)