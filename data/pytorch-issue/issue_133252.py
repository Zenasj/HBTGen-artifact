import torch

from transformers import AutoModelForCausalLM


torch_model = AutoModelForCausalLM.from_pretrained(
    "apple/OpenELM-270M-Instruct",
    torch_dtype=torch.float32,
    trust_remote_code=True,
    low_cpu_mem_usage=False,
    return_dict=False,
)
torch_model.eval()

example_inputs = (torch.zeros((1, 1), dtype=torch.int32),)
exported_program = torch.export.export(torch_model, example_inputs)