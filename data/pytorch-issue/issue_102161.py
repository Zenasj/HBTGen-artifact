import torch
import torch._dynamo
from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained("gpt2")
print(model.__class__)
model.to("cuda")
model.eval()

torch._dynamo.config.verbose = True
model = torch.compile(model, dynamic=True)
print(model.__class__)

inp_t = torch.randint(low=1, high=50255, size=[1, 1014], dtype=torch.long, device="cuda")
output_x = model(input_ids=inp_t)