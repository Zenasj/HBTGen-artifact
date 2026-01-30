python
from transformers import AutoModelForCausalLM
import peft
import torch

model = AutoModelForCausalLM.from_pretrained(
    "casperhansen/llama-3-8b-instruct-awq",
    device_map="auto",
)
model = peft.get_peft_model(
    model,
    peft.LoraConfig(
        task_type="CAUSAL_LM"
    )
)

torch._dynamo.config.cache_size_limit = 1024
for i, layer in enumerate(model.base_model.model.model.layers):
    model.base_model.model.model.layers[i] = torch.compile(layer)

with torch.amp.autocast("cuda"):
    model(
        input_ids = torch.tensor([[0, 1, 2]]).cuda(),
        attention_mask = torch.tensor([[1, 1, 1]]).cuda()
    )