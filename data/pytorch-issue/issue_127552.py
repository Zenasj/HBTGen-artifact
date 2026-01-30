import torch
import torch.nn as nn
import torch.nn.functional as F

if node.op == "call_module":
                target_mod = gm.get_submodule(node.target)
                if target_mod not in processed_modules:
                    self.add_module_params_to_bucket(
                        target_mod, buckets[0], processed_modules, node.target
                    )

{"dynamo_start": {"stack": "STACK"}, "rank": 0, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"dynamo_guards": {}, "rank": 0, "frame_id": 0, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"dynamo_cpp_guards_str": {}, "rank": 0, "frame_id": 0, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "rank": 0, "frame_id": 0, "frame_compile_id": 0, "attempt": 1}
{"dynamo_start": {"stack": "STACK"}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0}
{"dynamo_output_graph": {"sizes": {"l_self_layers_0_weight": [1024, 1024], "l_self_layers_0_bias": [1024], "l_x_": [1024, 1024], "l_self_layers_1_weight": [1024, 1024], "l_self_layers_1_bias": [1024], "input_1": [1024, 1024], "input_2": [1024, 1024]}}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"aot_joint_graph": {}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"aot_forward_graph": {}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"aot_backward_graph": {}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_post_grad_graph": {}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_output_code": {"filename": "FILENAME"}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"dynamo_guards": {}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"dynamo_cpp_guards_str": {}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0}

class GraphModule(torch.nn.Module):
    def forward(self, L_self_layers_0_weight: "f32[1024, 1024]", L_self_layers_0_bias: "f32[1024]", L_x_: "f32[1024, 1024]", L_self_layers_1_weight: "f32[1024, 1024]", L_self_layers_1_bias: "f32[1024]"):
        l_self_layers_0_weight = L_self_layers_0_weight
        l_self_layers_0_bias = L_self_layers_0_bias
        l_x_ = L_x_
        l_self_layers_1_weight = L_self_layers_1_weight
        l_self_layers_1_bias = L_self_layers_1_bias
        
        # File: /data/users/lsakka/pytorch/pytorch/torch/nn/modules/linear.py:116 in forward, code: return F.linear(input, self.weight, self.bias)
        input_1: "f32[1024, 1024]" = torch._C._nn.linear(l_x_, l_self_layers_0_weight, l_self_layers_0_bias);  l_x_ = l_self_layers_0_weight = l_self_layers_0_bias = None
        input_2: "f32[1024, 1024]" = torch._C._nn.linear(input_1, l_self_layers_1_weight, l_self_layers_1_bias);  input_1 = l_self_layers_1_weight = l_self_layers_1_bias = None
        return (input_2,)

class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[1024, 1024]"):
        l_x_ = L_x_
        
        # File: /data/users/lsakka/pytorch/pytorch/test/dynamo/test_structured_trace.py:284 in forward, code: return self.layers(x)
        l__self___layers_0: "f32[1024, 1024]" = self.L__self___layers_0(l_x_);  l_x_ = None
        l__self___layers_1: "f32[1024, 1024]" = self.L__self___layers_1(l__self___layers_0);  l__self___layers_0 = None
        return (l__self___layers_1,)