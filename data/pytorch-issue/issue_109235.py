import torch

from transformers import AutoTokenizer, T5ForConditionalGeneration

def input_example(self, max_batch=1, max_dim=64, seq_len=16):
    sample = next(self.parameters())
    input_ids = torch.randint(low=0, high=max_dim, size=(max_batch, seq_len), device=sample.device)
    labels = torch.randint(low=0, high=max_dim, size=(max_batch, seq_len), device=sample.device)
    attention_mask = torch.randint(low=0, high=1, size=(max_batch, seq_len), device=sample.device)
    return tuple([input_ids, attention_mask, labels])

model = T5ForConditionalGeneration.from_pretrained("google/byt5-small")

torch.onnx.export(model,
                  input_example(model),
                  "t5.onnx",
                  verbose=True,
                  opset_version=16,
                  do_constant_folding=True,
                  )