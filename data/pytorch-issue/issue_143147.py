import torch
import torch.nn as nn

class LayerNormPattern(torch.nn.Module):

    def forward(self, x: torch.Tensor, normalized_shape: List[int], 
                weight: torch.Tensor, bias: torch.Tensor, 
                eps: float) -> torch.Tensor:

        # match LayerNorm
        y = torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, eps)

        # ... optionally, match other functionality ...

        return y

@torch.library.custom_op("my_lib::my_layer_norm", mutates_args=())
def my_layer_norm(x: torch.Tensor, normalized_shape: List[int],
                  weight: torch.Tensor, bias: torch.Tensor, 
                  eps: float) -> torch.Tensor:

    # for now, returns corresponding ATEN operator output, but could contain a
    # customized version of layer_norm.
    y = torch.ops.aten.layer_norm(x, normalized_shape, weight, bias, eps)

    # ... other optional functionality ...

    return y

@torch.library.register_fake("my_lib::my_layer_norm")
def _(x: torch.Tensor, normalized_shape: List[int], 
      weight: torch.Tensor, bias:torch.Tensor, 
      eps: float):

    return torch.empty_like(x)

class LayerNormReplacement(torch.nn.Module):

    def forward(self, x: torch.Tensor, normalized_shape: List[int], 
                weight: torch.Tensor, bias: torch.Tensor, 
                eps: float) -> torch.Tensor:

        # match LayerNorm
        y = torch.ops.my_lib.my_layer_norm(x, normalized_shape, weight, bias, eps)

        # ... optionally, match other functionality ...

        return y

#
# Define the mapping between a node's arguments and its corresponding placeholder nodes.
#
LN_REPL_LITERAL_MAP = {
    'my_lib::my_layer_norm': {     # the node (call_function) for which this replacement applies
        1: 'normalized_shape',     # <node_arg_idx> -> <op_placeholder_name> mapping.
        4: 'eps',
    }
}

pattern_gm = torch.export.export(pattern_model, cls.example_inputs).module()
_replace_node_literals_with_existing_placeholders(pattern_gm, LN_PATTERN_LITERAL_MAP)