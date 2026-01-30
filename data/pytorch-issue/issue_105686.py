from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch._dynamo
from torch._inductor.compile_fx import compile_fx

model_path = "mosaicml/mpt-7b"

model = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
).eval()

tokenizer = AutoTokenizer.from_pretrained(
    model_path, trust_remote_code=True
)

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

    tokenized_inputs = tokenizer(
        "Testing PT2 Compile",
        return_tensors="pt",
    )

    print(tokenized_inputs)

    output = model.forward(
        input_ids=tokenized_inputs['input_ids'],
        attention_mask=tokenized_inputs['attention_mask']
    )

    print(output)

    output = compiled_forward(
        input_ids=tokenized_inputs['input_ids'],
        attention_mask=tokenized_inputs['attention_mask']
    )

    print(output)