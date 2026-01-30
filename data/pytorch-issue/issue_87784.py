import torch.nn as nn

import torch
import os
from transformers import AutoConfig, AutoModel
model = AutoModel.from_pretrained("bert-base-uncased")
model_quantized = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)  
quantized_output_dir = "quantized/"
if not os.path.exists(quantized_output_dir):
    os.makedirs(quantized_output_dir)
    model_quantized.save_pretrained(quantized_output_dir)

config.json

{
  "_name_or_path": "bert-base-uncased",
  "architectures": [
    "BertModel"
  ],
  "attention_probs_dropout_prob": 0.1,
  "classifier_dropout": null,
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
  "torch_dtype": "float32",
  "transformers_version": "4.23.1",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 30522
}