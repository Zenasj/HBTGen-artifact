import torch.nn as nn

...
 # File: /home/xadupre/.local/lib/python3.9/site-packages/transformers/models/llama/modeling_llama.py:654 in forward, code: value_states = value_states.contiguous()
clone_2: "f16[1, 32, 512, 128]" = torch.ops.aten.clone.default(transpose_3, memory_format = torch.contiguous_format)

 # File: /home/xadupre/.local/lib/python3.9/site-packages/transformers/models/llama/modeling_llama.py:660 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
unsqueeze_8: "f16[1, 512, 513]" = torch.ops.aten.unsqueeze.default(mul, 0)
unsqueeze_9: "f16[1, 1, 512, 513]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, 1);  unsqueeze_8 = None
slice_14: "f16[1, 1, 512, 513]" = torch.ops.aten.slice.Tensor(unsqueeze_9, 2, 0, 9223372036854775807);  unsqueeze_9 = None
slice_15: "f16[1, 1, 512, 513]" = torch.ops.aten.slice.Tensor(slice_14, 3, 0, 9223372036854775807);  slice_14 = None
expand_2: "f16[1, 1, 512, 513]" = torch.ops.aten.expand.default(slice_15, [1, 1, -1, -1]);  slice_15 = None
slice_16: "f16[1, 1, 512, 513]" = torch.ops.aten.slice.Tensor(expand_2, 0, 0, 9223372036854775807);  expand_2 = None
slice_17: "f16[1, 1, 512, 513]" = torch.ops.aten.slice.Tensor(slice_16, 1, 0, 9223372036854775807);  slice_16 = None
slice_18: "f16[1, 1, 512, 513]" = torch.ops.aten.slice.Tensor(slice_17, 2, 0, 9223372036854775807);  slice_17 = None
slice_19: "f16[1, 1, 512, 512]" = torch.ops.aten.slice.Tensor(slice_18, 3, 0, 512);  slice_18 = None
scaled_dot_product_attention: "f16[1, 32, 512, 128]" = torch.ops.aten.scaled_dot_product_attention.default(clone, clone_1, clone_2, slice_19);  clone = clone_1 = clone_2 = slice_19 = None

 # File: /home/xadupre/.local/lib/python3.9/site-packages/transformers/models/llama/modeling_llama.py:669 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
transpose_4: "f16[1, 512, 32, 128]" = torch.ops.aten.transpose.int(scaled_dot_product_attention, 1, 2);  scaled_dot_product_attention = None

 # File: /home/xadupre/.local/lib/python3.9/site-packages/transformers/models/llama/modeling_llama.py:670 in forward, code: attn_output = attn_output.view(bsz, q_len, -1)
view_4: "f16[1, 512, 4096]" = torch.ops.aten.view.default(transpose_4, [1, 512, -1]);  transpose_4 = None

 # File: /home/xadupre/.local/lib/python3.9/site-packages/transformers/models/llama/modeling_llama.py:672 in forward, code: attn_output = self.o_proj(attn_output)
linear_3: "f16[1, 512, 4096]" = torch.ops.aten.linear.default(view_4, p_model_model_layers_0_self_attn_o_proj_weight);  view_4 = p_model_model_layers_0_self_attn_o_proj_weight = None
...

def test_scaled_dot_product_attention(self):

        class DummyModel(torch.nn.Module):
            def __init__(self, enable_math: bool):
                super().__init__()
                self.enable_math = False

            def forward(self, query, key, value):
                res = torch.nn.functional.scaled_dot_product_attention(
                    query, key, value
                )
                rest = res.transpose(0, 1)
                final = rest.view(8, 32, 128 * 64)
                return final

        model = DummyModel(False)
        device = "cpu"

        query = torch.rand(32, 8, 128, 64, dtype=torch.float16, device=device)
        key = torch.rand(32, 8, 128, 64, dtype=torch.float16, device=device)
        value = torch.rand(32, 8, 128, 64, dtype=torch.float16, device=device)
        expected = model(query, key, value)
        self.assertEqual(expected.dtype, torch.float16)
        self.assertEqual(expected.shape, (8, 32, 8192))

        cpl = torch.compile(model)
        new_output = cpl(query, key, value)
        self.assertEqual(new_output.dtype, torch.float16)
        self.assertEqual(new_output.shape, (8, 32, 8192))
        self.assertEqualArray(expected, new_output)        

        export = torch.export.export(model, (query, key, value))
        # Next line fails here due to
        # Cannot view a tensor with shape torch.Size([8, 32, 128, 64]) and strides
        # (64, 512, 16384, 1) as a tensor with shape (8, 32, 8192)
        export.run_decompositions()

import torch

class Model(torch.nn.Module):

    def forward(self, query, key, value):
        res = torch.nn.functional.scaled_dot_product_attention(
            query, key, value
        )
        rest = res.transpose(0, 1)
        final = rest.view(8, 32, 128 * 64)
        return final

model = Model()
device = "cpu"

query = torch.rand(32, 8, 128, 64, dtype=torch.float16, device=device)
key = torch.rand(32, 8, 128, 64, dtype=torch.float16, device=device)
value = torch.rand(32, 8, 128, 64, dtype=torch.float16, device=device)

export = torch.export.export(model, (query, key, value))
# Next line fails here due to
# Cannot view a tensor with shape torch.Size([8, 32, 128, 64]) and strides
# (64, 512, 16384, 1) as a tensor with shape (8, 32, 8192)
export.run_decompositions()