import torch
import torch.nn as nn

class CompiledAutograd0(torch.nn.Module):
    def forward(self, inputs, sizes, scalars, hooks):
        ...
        aot0_fw_graph0 = self.fw_graph0
        aot0_joint_graph0 = self.joint_graph0
        aot0_mask_graph0 = self.mask_graph0
        aot0_flex_attention_backward = torch.ops.higher_order.flex_attention_backward(aot0_primals_1, aot0_primals_1, aot0_primals_1, aot0_detach_3, aot0_detach_5, aot0_expand_5, aot0_zeros_1, aot0_fw_graph0, aot0_joint_graph0, (aot0_ones, aot0_zeros, None, None, aot0__to_copy_1, aot0__to_copy_2, None, None, 1073741824, 1073741824, aot0_mask_graph0), 0.125, {'PRESCALE_QK': False, 'ROWS_GUARANTEED_SAFE': False, 'BLOCKS_ARE_CONTIGUOUS': False, 'OUTPUT_LOGSUMEXP': True}, (), ());  aot0_detach_3 = aot0_detach_5 = aot0_expand_5 = aot0_zeros_1 = aot0_fw_graph0 = aot0_joint_graph0 = aot0_ones = aot0_zeros = aot0__to_copy_1 = aot0__to_copy_2 = aot0_mask_graph0 = None
        aot0_getitem_4: "bf16[1, 1, s0, s1]" = aot0_flex_attention_backward[0]
        aot0_getitem_5: "bf16[1, 1, s0, s1]" = aot0_flex_attention_backward[1]
        aot0_getitem_6: "bf16[1, 1, s0, s1]" = aot0_flex_attention_backward[2];  aot0_flex_attention_backward = None
        ...