import torch
import torch.nn as nn

out_attn_bias = scaled_dot_product_attention(query, key, value, attn_mask=attn_bias, dropout_p=0.0)

var = tx.output.side_effects.track_object_new(
                self.source,
                self.value,
                variables.UnspecializedNNModuleVariable
                if issubclass(self.value, torch.nn.Module)
                else UserDefinedObjectVariable,
                {},
            )