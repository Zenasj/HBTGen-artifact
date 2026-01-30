from transformers import AutoModelForCausalLM
import torch
import torch._dynamo
from torch._inductor.compile_fx import compile_fx

model = AutoModelForCausalLM.from_pretrained(
    "mosaicml/mpt-7b",
    trust_remote_code=True,
).eval()

compiled_forward = torch._dynamo.optimize(
    lambda model, inputs: compile_fx(
        model,
        inputs,
        config_patches={
            "triton.cudagraphs": False,
            "size_asserts": False,
        },
    ),
    dynamic=True,
)(model.forward)

with torch.no_grad():

    one_token = torch.tensor([[1]])

    # uncomment this line and everything works?!
    # model.forward(input_ids=one_token, attention_mask=one_token)

    compiled_forward(input_ids=one_token, attention_mask=one_token)