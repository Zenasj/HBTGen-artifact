# torch.rand((512, 128), (128, 512), dtype=torch.float32) ← inferred input shape (tuple of two tensors)
import torch
from torch import nn
from torch._inductor.pattern_matcher import PatternMatcherPass, register_replacement, fwd_only

@torch.library.custom_op("neuralmagic::dual_mm", mutates_args=())
def dual_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    n = b.shape[1] // 2
    b_1 = b[(Ellipsis, slice(None, n, None))]
    b_2 = b[(Ellipsis, slice(n, None, None))]
    out_1 = a @ b_1
    out_2 = a @ b_2
    return out_1 + out_2

@dual_mm.register_fake
def dual_mm(a, b):
    return torch.empty(a.shape[0], b.shape[1] // 2)

def dual_mul_replacement(a, b):
    print("MATCHED")
    return torch.ops.neuralmagic.dual_mm(a, b)

def dual_mul_pattern(a, b):
    n = b.shape[1] // 2
    b_1 = b[(Ellipsis, slice(None, n, None))]
    b_2 = b[(Ellipsis, slice(n, None, None))]
    out_1 = a @ b_1
    out_2 = a @ b_2
    return out_1 + out_2

def custom_pass(graph):
    my_patterns = PatternMatcherPass()
    a = torch.empty((512, 128))
    b = torch.empty((512, 128)).t()
    register_replacement(dual_mul_pattern, dual_mul_replacement, [a, b], fwd_only, [my_patterns])
    my_patterns.apply(graph)

def test_backend(graph, example_inputs):
    from torch._inductor import config
    current_config = config.shallow_copy_dict()
    current_config['post_grad_custom_pre_pass'] = custom_pass
    from torch._inductor.compile_fx import compile_fx
    return compile_fx(graph, example_inputs, config_patches=current_config)

class MyModel(nn.Module):
    def forward(self, inputs):
        a, b = inputs
        n = b.shape[1] // 2
        b_1 = b[(Ellipsis, slice(None, n, None))]
        b_2 = b[(Ellipsis, slice(n, None, None))]
        out_1 = a @ b_1
        out_2 = a @ b_2
        return out_1 + out_2

def my_model_function():
    return MyModel()

def GetInput():
    m, k, n = 512, 128, 512
    a = torch.randn((m, k), dtype=torch.float32)
    b = torch.randn((n, k), dtype=torch.float32).t()
    torch._dynamo.mark_dynamic(b, 1)
    return (a, b)

