from transformers import LlamaTokenizer, LlamaForCausalLM
import torch
from peft import LoraConfig, get_peft_model, TaskType, get_peft_config

tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
model = LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf").to("cuda:6")
lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
model = get_peft_model(model, lora_config)

model = torch.compile(model)

# generate some dummy input

x = tokenizer("Hello, my dog is cute", return_tensors="pt")
x = x.to("cuda:6")
outputs = model(**x, labels=x["input_ids"])
outputs.loss.backward()