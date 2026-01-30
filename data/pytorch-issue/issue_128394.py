import torch
from torch.export._trace import _export
from transformers import AutoModelForCausalLM, AutoTokenizer


# Grab the model
# with torch.device("meta"):
llama = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf"
)
llama.eval()

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer.pad_token = tokenizer.eos_token

prompts = (
    "How do you", "I like to",
)

inputs = tokenizer(prompts, return_tensors="pt", padding=True)


ep = torch.export.export(
    llama,
    (inputs["input_ids"],),
    strict=True,
)