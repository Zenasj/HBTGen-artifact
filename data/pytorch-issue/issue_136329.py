import torch
from torch._inductor.pattern_matcher import PatternMatcherPass, register_replacement, fwd_only

torch.set_default_device("cuda")

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
    torch._dynamo.mark_dynamic(b, 1)
    register_replacement(dual_mul_pattern, 
                         dual_mul_replacement, 
                         [a, b], 
                         fwd_only, 
                         [my_patterns])
    my_patterns.apply(graph)
    print(f"AFTER: \n {graph}")
    print(f"CODE: \n {graph.python_code(graph.owning_module).src}")

def test_backend(graph, example_inputs):
    print(f"GRAPH: \n {graph}")
    from torch._inductor import config
    current_config = config.shallow_copy_dict()
    from torch._inductor.compile_fx import compile_fx
    current_config['post_grad_custom_post_pass'] = custom_pass
    return compile_fx(graph, example_inputs, config_patches=current_config)


@torch.compile(backend=test_backend)
def naive_dual_mul(a, b):
    n = b.shape[1] // 2
    b_1 = b[(Ellipsis, slice(None, n, None))]
    b_2 = b[(Ellipsis, slice(n, None, None))]

    out_1 = a @ b_1
    out_2 = a @ b_2
    return out_1 + out_2


m = 512
k = 128
n = 512

a = torch.randn((m, k))
b = torch.randn((n, k)).t()

torch._dynamo.mark_dynamic(b, 1)

c = naive_dual_mul(a, b)

import torch
from torch._inductor.pattern_matcher import PatternMatcherPass, register_replacement, fwd_only

torch.set_default_device("cuda")

k = 128
n = 512

@torch.library.custom_op("neuralmagic::half_slice", mutates_args=())
def half_slice(b: torch.Tensor) -> torch.Tensor:
    n = 256
    b_1 = b[(Ellipsis, slice(None, n, None))]
    return b_1.clone()

@half_slice.register_fake
def half_slice(b): 
    return torch.empty(b.shape[0], b.shape[1] // 2)

def slice_replacement(b):
    print("MATCH")
    return torch.ops.neuralmagic.half_slice(b)

def slice_pattern(b):
    n = 256
    b_1 = b[(Ellipsis, slice(None, n, None))]
    return b_1.clone()

b = torch.empty((n, k)).t()

my_patterns = PatternMatcherPass()
register_replacement(slice_pattern, 
                     slice_replacement, 
                     [b], 
                     fwd_only, 
                     [my_patterns])

def custom_pass(graph):
    my_patterns.apply(graph)

def test_backend(graph, example_inputs):
    graph.print_readable()
    from torch._inductor import config
    current_config = config.shallow_copy_dict()
    from torch._inductor.compile_fx import compile_fx
    current_config['post_grad_custom_post_pass'] = custom_pass
    return compile_fx(graph, example_inputs, config_patches=current_config)


@torch.compile(backend=test_backend)
def naive_slice(b):
    n = 256
    b_1 = b[(Ellipsis, slice(None, n, None))]
    return b_1.clone()

b = torch.randn((n, k)).t()
c = naive_slice(b)