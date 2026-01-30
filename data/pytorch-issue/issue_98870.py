import torch

from torch._subclasses.fake_tensor import FakeTensorMode

a = 1
b = torch.ones([10])
print(a in b) # this works and evaluated to be True

with FakeTensorMode():
    a = torch.SymInt(1)
    b = torch.ones([10])
    a in b # failed

with FakeTensorMode():
    a = 1
    b = torch.ones([10])
    a in b # failed

from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch

model_id = "EleutherAI/gpt-neo-125M"
model = GPTNeoForCausalLM.from_pretrained(model_id)
tokenizer = GPT2Tokenizer.from_pretrained(model_id)

prompt = (
    "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
    "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
    "researchers was the fact that the unicorns spoke perfect English."
)

input_ids = tokenizer(prompt, return_tensors="pt").input_ids


@torch._dynamo.optimize(dynamic=True)
def predict():
    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.9,
        max_length=100,
    )
    return tokenizer.batch_decode(gen_tokens)[0]

predict()