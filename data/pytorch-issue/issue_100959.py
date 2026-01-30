import torch
import torch.nn as nn
import copy
from torch.ao.quantization import (
    FusedMovingAvgObsFakeQuantize,
    observer,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    QConfigMapping,
)
from torch.ao.quantization._pt2e.quantizer import (
    OperatorConfig,
    QNNPackQuantizer,
    Quantizer,
)
import operator
from typing import Any, List, Tuple
import torch._dynamo as torchdynamo
from torch.ao.quantization._quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e_quantizer,
)


def test():
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.conv = nn.Conv2d(2, 2, 1)
                self.pool = nn.MaxPool2d(1, 1)

            def forward(self, x):
                x = self.conv(x)
                x = self.pool(x)
                return x

        class BackendAQuantizer(Quantizer):
            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                _DEFAULT_TARGET_DTYPE_INFO = {
                    "input_act_obs_or_fq_ctr": observer.PlaceholderObserver.with_args(
                        dtype=torch.float
                    ),
                    "output_act_obs_or_fq_ctr": observer.PlaceholderObserver.with_args(
                        dtype=torch.float
                    ),
                }
                for node in model.graph.nodes:
                    node.meta["target_dtype_info"] = copy.deepcopy(
                        _DEFAULT_TARGET_DTYPE_INFO
                    )
                for node in model.graph.nodes:
                    if (
                        node.op == "call_function"
                        and node.target == torch.ops.aten.convolution.default
                    ):
                        node.meta["target_dtype_info"] = {
                            "input_act_obs_or_fq_ctr": observer.default_observer,
                            "weight_obs_or_fq_ctr": observer.default_weight_observer,
                            "bias_obs_or_fq_ctr": observer.PlaceholderObserver.with_args(
                                dtype=torch.float
                            ),
                            "weight_index": 1,
                            "bias_index": 2,
                        }
                    if (
                        node.op == "call_function"
                        and node.target == operator.getitem
                        and node.args[1] == 0
                    ):
                        getitem_node = node
                        maxpool_node = getitem_node.args[0]
                        maxpool_node.meta["target_dtype_info"] = {
                            "input_act_obs_or_fq_ctr":observer.default_observer,
                            "_annotated": True,
                        }
                        getitem_node.meta["target_dtype_info"] = {
                            "output_act_obs_or_fq_ctr": observer.default_observer,
                            "input_output_share_observers": True,
                            "_annotated": True,
                        }

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

            @classmethod
            def get_supported_operators(cls) -> List[OperatorConfig]:
                pass        

        m = M()
        m_pt2e = copy.deepcopy(m)
        x = torch.rand(1, 2, 14, 14)
        example_inputs = (x,)
        # program capture
        m, guards = torchdynamo.export(
            m,
            *copy.deepcopy(example_inputs),
            aten_graph=True,
        )
        m = prepare_pt2e_quantizer(m, BackendAQuantizer())
        m(*example_inputs)
        m = convert_pt2e(m)
        print("m after convert is: {}".format(m), flush=True)

if __name__ == "__main__":
   test()

def forward(self, x):
    arg0, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    _scale_0 = self._scale_0
    _zero_point_0 = self._zero_point_0
    quantize_per_tensor = torch.ops.quantized_decomposed.quantize_per_tensor(arg0, _scale_0, _zero_point_0, 0, 127, torch.uint8);  arg0 = None
    dequantize_per_tensor = torch.ops.quantized_decomposed.dequantize_per_tensor(quantize_per_tensor, _scale_0, _zero_point_0, 0, 127, torch.uint8);  quantize_per_tensor = _scale_0 = _zero_point_0 = None
    _param_constant0 = self._param_constant0
    conv_scale_0 = self.conv_scale_0
    conv_zero_point_0 = self.conv_zero_point_0
    quantize_per_tensor_1 = torch.ops.quantized_decomposed.quantize_per_tensor(_param_constant0, conv_scale_0, conv_zero_point_0, -128, 127, torch.int8);  _param_constant0 = None
    dequantize_per_tensor_1 = torch.ops.quantized_decomposed.dequantize_per_tensor(quantize_per_tensor_1, conv_scale_0, conv_zero_point_0, -128, 127, torch.int8);  quantize_per_tensor_1 = conv_scale_0 = conv_zero_point_0 = None
    _param_constant1 = self._param_constant1
    convolution_default = torch.ops.aten.convolution.default(dequantize_per_tensor, dequantize_per_tensor_1, _param_constant1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  dequantize_per_tensor = dequantize_per_tensor_1 = _param_constant1 = None
    max_pool2d_with_indices_default = torch.ops.aten.max_pool2d_with_indices.default(convolution_default, [1, 1], [1, 1]);  convolution_default = None
    getitem = max_pool2d_with_indices_default[0];  max_pool2d_with_indices_default = None
    pool_scale_0 = self.pool_scale_0
    pool_zero_point_0 = self.pool_zero_point_0
    quantize_per_tensor_2 = torch.ops.quantized_decomposed.quantize_per_tensor(getitem, pool_scale_0, pool_zero_point_0, 0, 127, torch.uint8);  getitem = None
    dequantize_per_tensor_2 = torch.ops.quantized_decomposed.dequantize_per_tensor(quantize_per_tensor_2, pool_scale_0, pool_zero_point_0, 0, 127, torch.uint8);  quantize_per_tensor_2 = pool_scale_0 = pool_zero_point_0 = None
    return pytree.tree_unflatten([dequantize_per_tensor_2], self._out_spec)

def forward(self, x):
    arg0, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    _scale_0 = self._scale_0
    _zero_point_0 = self._zero_point_0
    quantize_per_tensor = torch.ops.quantized_decomposed.quantize_per_tensor(arg0, _scale_0, _zero_point_0, 0, 127, torch.uint8);  arg0 = None
    dequantize_per_tensor = torch.ops.quantized_decomposed.dequantize_per_tensor(quantize_per_tensor, _scale_0, _zero_point_0, 0, 127, torch.uint8);  quantize_per_tensor = _scale_0 = _zero_point_0 = None
    _param_constant0 = self._param_constant0
    conv_scale_0 = self.conv_scale_0
    conv_zero_point_0 = self.conv_zero_point_0
    quantize_per_tensor_1 = torch.ops.quantized_decomposed.quantize_per_tensor(_param_constant0, conv_scale_0, conv_zero_point_0, -128, 127, torch.int8);  _param_constant0 = None
    dequantize_per_tensor_1 = torch.ops.quantized_decomposed.dequantize_per_tensor(quantize_per_tensor_1, conv_scale_0, conv_zero_point_0, -128, 127, torch.int8);  quantize_per_tensor_1 = conv_scale_0 = conv_zero_point_0 = None
    _param_constant1 = self._param_constant1
    convolution_default = torch.ops.aten.convolution.default(dequantize_per_tensor, dequantize_per_tensor_1, _param_constant1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  dequantize_per_tensor = dequantize_per_tensor_1 = _param_constant1 = None
    conv_scale_1 = self.conv_scale_1
    conv_zero_point_1 = self.conv_zero_point_1
    quantize_per_tensor_2 = torch.ops.quantized_decomposed.quantize_per_tensor(convolution_default, conv_scale_1, conv_zero_point_1, 0, 127, torch.uint8);  convolution_default = None
    dequantize_per_tensor_2 = torch.ops.quantized_decomposed.dequantize_per_tensor(quantize_per_tensor_2, conv_scale_1, conv_zero_point_1, 0, 127, torch.uint8);  quantize_per_tensor_2 = conv_scale_1 = conv_zero_point_1 = None
    max_pool2d_with_indices_default = torch.ops.aten.max_pool2d_with_indices.default(dequantize_per_tensor_2, [1, 1], [1, 1]);  dequantize_per_tensor_2 = None
    getitem = max_pool2d_with_indices_default[0];  max_pool2d_with_indices_default = None
    pool_scale_0 = self.pool_scale_0
    pool_zero_point_0 = self.pool_zero_point_0
    quantize_per_tensor_3 = torch.ops.quantized_decomposed.quantize_per_tensor(getitem, pool_scale_0, pool_zero_point_0, 0, 127, torch.uint8);  getitem = None
    dequantize_per_tensor_3 = torch.ops.quantized_decomposed.dequantize_per_tensor(quantize_per_tensor_3, pool_scale_0, pool_zero_point_0, 0, 127, torch.uint8);  quantize_per_tensor_3 = pool_scale_0 = pool_zero_point_0 = None
    return pytree.tree_unflatten([dequantize_per_tensor_3], self._out_spec)