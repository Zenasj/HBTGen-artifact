import torch.nn as nn

input_names
# 'input_ids', 'past_key_values.0.key', 'past_key_values.0.value', 'past_key_values.1.key', 'past_key_values.1.value', 'past_key_values.2.key', 'past_key_values.2.value', 'past_key_values.3.key', 'past_key_values.3.value', 'past_key_values.4.key', 'past_key_values.4.value', 'attention_mask', 'position_ids'

inspect.sig(model.forward).parameters
# mappingproxy(OrderedDict([('input_ids', <Parameter "input_ids: Optional[torch.LongTensor] = None">), ('past_key_values', <Parameter "past_key_values: Union[transformers.cache_utils.Cache, Tuple[Tuple[torch.Tensor]], NoneType] = None">), ('attention_mask', <Parameter "attention_mask: Optional[torch.FloatTensor] = None">), ('token_type_ids', <Parameter "token_type_ids: Optional[torch.LongTensor] = None">), ('position_ids', <Parameter "position_ids: Optional[torch.LongTensor] = None">), ('head_mask', <Parameter "head_mask: Optional[torch.FloatTensor] = None">), ('inputs_embeds', <Parameter "inputs_embeds: Optional[torch.FloatTensor] = None">), ('labels', <Parameter "labels: Optional[torch.LongTensor] = None">), ('use_cache', <Parameter "use_cache: Optional[bool] = None">), ('output_attentions', <Parameter "output_attentions: Optional[bool] = None">), ('output_hidden_states', <Parameter "output_hidden_states: Optional[bool] = None">), ('return_dict', <Parameter "return_dict: Optional[bool] = None">), ('cache_position', <Parameter "cache_position: Optional[torch.LongTensor] = None">)]))

import torch


class Model(torch.nn.Module):
    def forward(self, x=None, y=None):
        return x + y

dim = torch.export.Dim("x", min=1, max=6)
onnx_program = torch.export.export(
    Model(),
    (),
    kwargs={"x": torch.randn(2, 3), "y": torch.randn(2, 3)},
    dynamic_shapes={"custom_input_x": {0: dim}, "custom_input_y": {0: dim}},
)

# torch._dynamo.exc.UserError: When `dynamic_shapes` is specified as a dict, its top-level keys must be the arg names ['x', 'y'] of `inputs`, but here they are ['custom_input_x', 'custom_input_y']. Alternatively, you could also ignore arg names entirely and specify `dynamic_shapes` as a list/tuple matching `inputs`. For more information about this error, see: https://pytorch.org/docs/main/generated/exportdb/index.html#dynamic-shapes-validation