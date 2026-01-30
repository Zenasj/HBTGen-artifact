py
import torch
import transformers
import logging
torch._logging.set_logs(dynamo=logging.INFO,inductor=logging.INFO)


device = torch.device("cuda:0")
model = transformers.BertForMaskedLM.from_pretrained("bert-base-uncased")

model.eval()
model.cuda()
model.half()

bs = 1
ins = {'input_ids': torch.randint(0, 10, size=(bs, 512)).to(device), 'attention_mask': torch.ones(bs, 512, dtype=torch.int64).to(device)}

with torch.no_grad():
    module, tmp = torch._dynamo.export(model,  **ins)

def specializations(input_ids: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None, token_type_ids: Optional[torch.Tensor] = None, position_ids: Optional[torch.Tensor] = None, head_mask: Optional[torch.Tensor] = None, inputs_embeds: Optional[torch.Tensor] = None, encoder_hidden_states: Optional[torch.Tensor] = None, encoder_attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None):
    # input_ids:
    assert input_ids.size()[0] == 1
    assert input_ids.size()[1] == 512

    # attention_mask:
    assert attention_mask.size()[0] == 1
    assert attention_mask.size()[1] == 512

py
import torch.utils._pytree as pytree
from transformers.modeling_outputs import MaskedLMOutput

def _flatten(d):
    return d.to_tuple(), None

def _unflatten(values, _):
    return MaskedLMOutput(*values)

pytree._register_pytree_node(MaskedLMOutput, _flatten, _unflatten)