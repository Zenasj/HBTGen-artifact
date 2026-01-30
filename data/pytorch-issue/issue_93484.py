import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sentence = "Question: Can I run BLOOM on a single GPU? Answer:"

# Load model
def load_model(model_name: str = "bigscience/bloom-560m"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        offload_state_dict=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    inputs = tokenizer(sentence, return_tensors="pt").to(0)
    print(inputs.keys())
    return model, inputs, tokenizer


# Inference in PyTorch
def run_model(model, inputs, tokenizer):
    with torch.no_grad():
        outputs = model(**inputs, return_dict=False)

    token_id = outputs[0][0][-1].argmax()
    answer = tokenizer.decode([token_id])
    print(f"{sentence}\n{answer}")


# Inference in dynamo
def run_dynamo(model, inputs, tokenizer):
    from torch import _dynamo as torchdynamo
    opt_model = torchdynamo.optimize("eager")(model)
    run_model(opt_model, inputs, tokenizer)


model, inputs, tokenizer = load_model()
run_model(model, inputs, tokenizer)  # this works
run_dynamo(model, inputs, tokenizer)  # this fails