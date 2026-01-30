import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

@torch.compile(backend="inductor")
def test_gpt2_demo():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(torch.float)
    prompt = "Thanks for"
    print("\nInput prompt: ", prompt, "\n")

    # run on CPU
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    gen_tokens = model.generate(
        input_ids,
        max_length=4,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_text = tokenizer.batch_decode(gen_tokens)[0]

    print("CPU output: ", gen_text, "\n")

test_gpt2_demo()

PretrainedConfig.pruned_heads

getattr

model.generate

torch.compile