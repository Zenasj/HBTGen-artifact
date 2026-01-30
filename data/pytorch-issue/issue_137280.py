from typing import List, Callable, Optional, Tuple

import torch
from torch._inductor.pattern_matcher import PatternMatcherPass, register_replacement, fwd_only
from torch._higher_order_ops.auto_functionalize import auto_functionalized

torch.set_default_device("cuda")


@torch.library.custom_op("vllm::fused_add_rms_norm_quant_static", mutates_args=['result', 'residual'])
def fused_add_rms_norm_quant_static(result: torch.Tensor, input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor,
                                azp: torch.Tensor, epsilon: float) -> None:
    print("vllm::fused_rms_norm_quant_static")
    # bogus implementation doesn't matter
    result_rms = residual + torch.mul(input, weight) + epsilon
    residual = input + residual
    result = torch.mul(result_rms, scale).to(torch.int8)
    scale.fill_(0.5)


@torch.library.register_fake("vllm::fused_add_rms_norm_quant_static")
def _(result: torch.Tensor, input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor,
      azp: torch.Tensor, epsilon: float) -> None:
    return None


@torch.library.custom_op("vllm::fused_add_rms_norm", mutates_args=['input', 'residual'])
def fused_add_rms_norm(input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, epsilon: float) -> None:
    # bogus implementation doesn't matter
    result = residual + torch.mul(input, weight) + epsilon
    residual = input + residual


@torch.library.register_fake("vllm::fused_add_rms_norm")
def _(input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, epsilon: float) -> None:
    return None


@torch.library.custom_op("vllm::static_scaled_int8_quant", mutates_args=['result'])
def static_scaled_int8_quant(result: torch.Tensor,
                             input: torch.Tensor,
                             scale: torch.Tensor,
                             azp: Optional[torch.Tensor] = None) -> None:
    # bogus implementation doesn't matter
    result = torch.mul(input, scale).to(torch.int8)
    # scale.fill_(0.5)


@torch.library.register_fake("vllm::static_scaled_int8_quant")
def _(result: torch.Tensor, input: torch.Tensor, scale: torch.Tensor, azp: Optional[torch.Tensor] = None) -> None:
    return None

def add_rms_pattern_static(result: torch.Tensor, input: torch.Tensor,
                    residual: torch.Tensor, weight: torch.Tensor,
                    scale: torch.Tensor):
    at1 = auto_functionalized(torch.ops.vllm.fused_add_rms_norm.default,
                             input=input,
                             residual=residual,
                             weight=weight,
                             epsilon=1e-6)
    at2 = auto_functionalized(torch.ops.vllm.static_scaled_int8_quant.default,
                              result=result,
                              input=at1[1],
                              scale=scale,
                              azp=None)

    # result, residual
    return at2[1], at1[2]
    # doesn't work with this order either
    # return at1[2], at2[1]

def add_rms_replacement_static(result: torch.Tensor, input: torch.Tensor,
                residual: torch.Tensor, weight: torch.Tensor,
                scale: torch.Tensor):
    at = auto_functionalized(
        torch.ops.vllm.fused_add_rms_norm_quant_static.default,
        result=result,
        input=input,
        residual=residual,
        weight=weight,
        scale=scale,
        azp=None,
        epsilon=1e-6)

    # result, residual
    return at[1], at[2]
    # doesn't work with this order either
    # return at[2], at[1]


def empty_bf16(*args, **kwargs):
    return torch.empty(*args, **kwargs, dtype=torch.bfloat16)


def empty_int8(*args, **kwargs):
    return torch.empty(*args, **kwargs, dtype=torch.int8)


my_patterns = PatternMatcherPass()
inputs = [empty_int8(5, 4), empty_bf16(5, 4), empty_bf16(5, 4), empty_bf16(5, 1), torch.empty(1, 1)]
register_replacement(add_rms_pattern_static, add_rms_replacement_static, inputs, fwd_only, my_patterns)

def custom_pass(graph: torch.fx.Graph) -> torch.fx.Graph:
    print(graph.python_code(root_module="self", verbose=True).src, end="\n\n")
    count = my_patterns.apply(graph)
    print(f"Count: {count}")
    graph.eliminate_dead_code()
    print(graph.python_code(root_module="self", verbose=True).src, end="\n\n")
    return graph

def custom_backend(graph: torch.fx.GraphModule, example_inputs: List[torch.Tensor]) -> Callable:
    from torch._inductor import config
    current_config = config.shallow_copy_dict()
    from torch._inductor.compile_fx import compile_fx
    current_config['post_grad_custom_post_pass'] = custom_pass
    return compile_fx(graph, example_inputs, config_patches=current_config)

@torch.compile(backend=custom_backend)
def my_func_static(x, w, epsilon):
    x2 = x.clone()
    residual = torch.relu(x2)
    torch.ops.vllm.fused_add_rms_norm(x2, residual, w, epsilon)
    scale = torch.ones((1, 1))
    quant_result = torch.empty_like(x2, dtype=torch.int8)
    torch.ops.vllm.static_scaled_int8_quant(quant_result, x2, scale, None)
    return quant_result, residual


print("Run my_func_static")
inputs = [empty_bf16((5, 4)), empty_bf16((5, 1)), 1e-6]
print(my_func_static(*inputs))