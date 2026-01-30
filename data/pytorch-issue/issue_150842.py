import torch
import json
from transformers import BertModel, BertConfig
import os

CONFIG = """
{
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "transformers_version": "4.6.0.dev0",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 30522
}
"""

config = json.loads(CONFIG)
bloom_config = BertConfig(**config)
model = BertModel(bloom_config).half().cuda()

vocab_size = 30522

input_ids = torch.randint(0, vocab_size, (2, 3)).cuda()
attention_mask = torch.ones(2, 3).cuda()
example_inputs = (input_ids, attention_mask)
batch_dim = torch.export.Dim("batch", min = 2, max = 10)
s_dim = torch.export.Dim("s")
exported = torch.export.export(model, example_inputs, 
                               dynamic_shapes = {"input_ids": {0: batch_dim, 1: torch.export.Dim.STATIC},
                                                 "attention_mask": {0:  torch.export.Dim.STATIC, 
                                                                    1:  torch.export.Dim.STATIC}})