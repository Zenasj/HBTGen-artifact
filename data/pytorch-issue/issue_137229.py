def is_mutation_op(node: torch.fx.Node) -> bool:
    if node.op == "call_function":
        if _mutation_op_re.search(node.target.__name__):  # type: ignore[union-attr]                                                                             
            return True
    elif node.op == "call_method":
        if _mutation_op_re.search(node.target):  # type: ignore[union-attr, arg-type]                                                                            
            return True
    return node.kwargs.get("out") is not None

from typing import List, Callable, Optional, Tuple

import torch
from torch._inductor.pattern_matcher import PatternMatcherPass, register_replacement, fwd_only
from torch._higher_order_ops.auto_functionalize import auto_functionalized

torch.set_default_device("cuda")


@torch.library.custom_op("vllm::fused_rms_norm_quant_static", mutates_args=['out'])
def fused_rms_norm_quant_static(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor,
                                azp: torch.Tensor, epsilon: float) -> None:
    print("vllm::fused_rms_norm_quant_static")
    # bogus implementation doesn't matter
    result_rms = torch.mul(input, weight) + epsilon
    out = torch.mul(result_rms, scale).to(torch.int8)


@torch.library.custom_op("vllm::rms_norm", mutates_args=['out'])
def rms_norm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor,
              epsilon: float) -> None:
    # bogus implementation doesn't matter
    out = torch.mul(input, weight) + epsilon


@torch.library.custom_op("vllm::static_scaled_int8_quant", mutates_args=['out'])
def static_scaled_int8_quant(out: torch.Tensor,
                             input: torch.Tensor,
                             scale: torch.Tensor,
                             azp: Optional[torch.Tensor] = None) -> None:
    # bogus implementation doesn't matter
    out = torch.mul(input, scale).to(torch.int8)


def rms_pattern_static(out: torch.Tensor, result_rms: torch.Tensor, input: torch.Tensor, weight: torch.Tensor,
                       scale: torch.Tensor):
    at1 = auto_functionalized(torch.ops.vllm.rms_norm.default, out=result_rms, input=input, weight=weight, epsilon=1e-6)
    at2 = auto_functionalized(torch.ops.vllm.static_scaled_int8_quant.default, out=out, input=at1[1], scale=scale,
                              azp=None)

    # result
    return at2[1]


def rms_replacement_static(result: torch.Tensor, result_rms: torch.Tensor, input: torch.Tensor, weight: torch.Tensor,
                           scale: torch.Tensor):
    at = auto_functionalized(torch.ops.vllm.fused_rms_norm_quant_static.default, out=result, input=input, weight=weight,
                             epsilon=1e-6, scale=scale, azp=None)

    # result
    return at[1]

def empty_bf16(*args, **kwargs):
    return torch.empty(*args, **kwargs, dtype=torch.bfloat16)


def empty_int8(*args, **kwargs):
    return torch.empty(*args, **kwargs, dtype=torch.int8)


my_patterns = PatternMatcherPass()
inputs = [empty_int8(5, 4), empty_bf16(5, 4), empty_bf16(5, 4), empty_bf16(5, 1), torch.empty(1, 1)]
register_replacement(rms_pattern_static, rms_replacement_static, inputs, fwd_only, my_patterns)

def custom_pass(graph: torch.fx.Graph) -> torch.fx.Graph:
    #print("Pre-pass:")
    #print(graph)
    count = my_patterns.apply(graph)
    print(f"Count: {count}")
    #print(graph)
    return graph

def custom_backend(graph: torch.fx.GraphModule, example_inputs: List[torch.Tensor]) -> Callable:
    from torch._inductor import config
    current_config = config.shallow_copy_dict()
    from torch._inductor.compile_fx import compile_fx
    current_config['post_grad_custom_post_pass'] = custom_pass
    return compile_fx(graph, example_inputs, config_patches=current_config)

@torch.compile(backend=custom_backend)
def my_func_static(x, w, epsilon):
    result = torch.empty_like(x)
    torch.ops.vllm.rms_norm(result, x, w, epsilon)
    scale = torch.ones((1, 1))
    quant_result = torch.empty_like(x, dtype=torch.int8)
    torch.ops.vllm.static_scaled_int8_quant(quant_result, result, scale, None)
    return quant_result


print("Run my_func_static")
inputs = [torch.empty((5, 4)), torch.empty((5, 1)), 1e-6]
my_func_static(*inputs)

from typing import List, Callable, Optional, Tuple

import torch
from torch._inductor.pattern_matcher import PatternMatcherPass, register_replacement, fwd_only
from torch._higher_order_ops.auto_functionalize import auto_functionalized

torch.set_default_device("cuda")


@torch.library.custom_op("vllm::fused_rms_norm_quant_static", mutates_args=['result'])
def fused_rms_norm_quant_static(result: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor,
                                azp: torch.Tensor, epsilon: float) -> None:
    print("vllm::fused_rms_norm_quant_static")
    # bogus implementation doesn't matter
    result_rms = torch.mul(input, weight) + epsilon
    result = torch.mul(result_rms, scale).to(torch.int8)


@torch.library.custom_op("vllm::rms_norm", mutates_args=['result'])
def rms_norm(result: torch.Tensor, input: torch.Tensor, weight: torch.Tensor,
              epsilon: float) -> None:
    # bogus implementation doesn't matter
    result = torch.mul(input, weight) + epsilon


@torch.library.custom_op("vllm::static_scaled_int8_quant", mutates_args=['result'])
def static_scaled_int8_quant(result: torch.Tensor,
                              input: torch.Tensor,
                              scale: torch.Tensor,
                              azp: Optional[torch.Tensor] = None) -> None:
    # bogus implementation doesn't matter
    result = torch.mul(input, scale).to(torch.int8)


def rms_pattern_static(result: torch.Tensor, result_rms: torch.Tensor, input: torch.Tensor, weight: torch.Tensor,
                       scale: torch.Tensor):
    at1 = auto_functionalized(torch.ops.vllm.rms_norm.default, result=result_rms, input=input, weight=weight, epsilon=1e-6)
    at2 = auto_functionalized(torch.ops.vllm.static_scaled_int8_quant.default, result=result, input=at1[1], scale=scale,
                              azp=None)

    # result
    return at2[1]


def rms_replacement_static(result: torch.Tensor, result_rms: torch.Tensor, input: torch.Tensor, weight: torch.Tensor,
                           scale: torch.Tensor):
    at = auto_functionalized(torch.ops.vllm.fused_rms_norm_quant_static.default, result=result, input=input, weight=weight,
                             epsilon=1e-6, scale=scale, azp=None)

    # result
    return at[1]


def empty_bf16(*args, **kwargs):
    return torch.empty(*args, **kwargs, dtype=torch.bfloat16)


def empty_int8(*args, **kwargs):
    return torch.empty(*args, **kwargs, dtype=torch.int8)


my_patterns = PatternMatcherPass()
inputs = [empty_int8(5, 4), empty_bf16(5, 4), empty_bf16(5, 4), empty_bf16(5, 1), torch.empty(1, 1)]
register_replacement(rms_pattern_static, rms_replacement_static, inputs, fwd_only, my_patterns)

def custom_pass(graph: torch.fx.Graph) -> torch.fx.Graph:
    #print("Pre-pass:")
    #print(graph)
    count = my_patterns.apply(graph)
    print(f"Count: {count}")
    #print(graph)
    return graph

def custom_backend(graph: torch.fx.GraphModule, example_inputs: List[torch.Tensor]) -> Callable:
    from torch._inductor import config
    current_config = config.shallow_copy_dict()
    from torch._inductor.compile_fx import compile_fx
    current_config['post_grad_custom_post_pass'] = custom_pass
    return compile_fx(graph, example_inputs, config_patches=current_config)

@torch.compile(backend=custom_backend)
def my_func_static(x, w, epsilon):
    result = torch.empty_like(x)
    torch.ops.vllm.rms_norm(result, x, w, epsilon)
    scale = torch.ones((1, 1))
    quant_result = torch.empty_like(x, dtype=torch.int8)
    torch.ops.vllm.static_scaled_int8_quant(quant_result, result, scale, None)
    return quant_result


print("Run my_func_static")
inputs = [torch.empty((5, 4)), torch.empty((5, 1)), 1e-6]
my_func_static(*inputs)