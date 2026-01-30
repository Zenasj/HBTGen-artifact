# torch/distributed/_composable/fsdp/_fsdp_common.py
def compiled_autograd_enabled():
    if torch.compiler.is_compiling():
        import torch._dynamo.compiled_autograd as ca
    
        print("graph break")  # add a graph break
        return ca.compiled_autograd_enabled or ca.in_compiled_autograd_region
    else:
        return False