import torch

checkpoint = 'databricks/dolly-v1-6b'
model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
global_pruning(linear_layers_list, prune_percentage=prune_percentage)

def global_pruning(linear_layers_list, prune_percentage):
    parameters_to_prune = tuple((x, 'weight') for x in linear_layers_list)
    prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=prune_percentage)