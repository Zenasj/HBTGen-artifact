3
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model = "Salesforce/codegen-350M-mono"
device = torch.device("mps")

# Load the CodeGen tokenizer and model
print("[+] Initializing Tokenizer")
tokenizer = AutoTokenizer.from_pretrained(model)
print("[+] Finished Initializing Tokenizer")
print("[+] Initializing Model")
model = AutoModelForCausalLM.from_pretrained(model).to(device)
print("[+] Finished Initializing Model")

prompt = "write a hello world function"

# Tokenize the description and generate the code
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    inputs["input_ids"].to('mps'),
    attention_mask = inputs["attention_mask"].to(device),
    min_new_tokens = 10,
    max_new_tokens = 1000,
    pad_token_id = 0,
    eos_token_id = tokenizer.eos_token_id,
    do_sample = True)

generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_code)