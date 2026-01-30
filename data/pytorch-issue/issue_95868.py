import torch
import torch.nn as nn
from transformers import AutoModel, BertConfig

class MyBert(nn.Module):
  def __init__(self):
    super().__init__()
    config = BertConfig.from_pretrained(
      "config.json")
    self.bert = AutoModel.from_pretrained(
      "model.pt", config=config)
    for param in self.bert.parameters():
      param.requires_grad_(False)
    d_model = self.bert.config.to_dict()['hidden_size']
    self.transform = nn.TransformerEncoderLayer(
      d_model=d_model,
      nhead=8,
      dim_feedforward=2048,
      batch_first=True
    )

  def forward(self, x):
    bert_out = self.bert(**x)
    x = self.transform(
      bert_out.last_hidden_state,
      src_key_padding_mask=None)
    return x

if __name__ == "__main__":
  input_example = {
    "input_ids": torch.randint(100, 5000, (1, 76)),
    "token_type_ids": torch.zeros(1, 76).long(),
    "attention_mask": torch.ones(1, 76).long()
  }
  model = MyBert()
  out = model(input_example)
  import torch._dynamo.config
  from torch._dynamo import optimize
  torch._dynamo.config.dynamic_shapes = True
  opt_model = optimize("inductor")(model)
  opt_out = opt_model(input_example)
  print(len(opt_out))