import torch
from transformers import TapasForQuestionAnswering

model_name = "google/tapas-base-finetuned-wtq"
model = TapasForQuestionAnswering.from_pretrained(model_name, torchscript=True)

bs = 1
seq_len = 512
dummy_inputs = (
    torch.ones(bs, seq_len, dtype=torch.long),
    torch.ones(bs, seq_len),
    torch.zeros(bs, seq_len, 7, dtype=torch.long),
)

torch.onnx.export(
    model,
    dummy_inputs,
    "scatter.onnx",
    export_params=True,
    opset_version=13,
    input_names=["input_ids", "attention_mask", "token_type_ids"],
    output_names=["logits", "logits_aggregation"],
    dynamic_axes={
        "input_ids": [0, 1],
        "token_type_ids": [0, 1],
        "attention_mask": [0, 1],
        "logits": [0, 1],
        "logits_aggregation": [0, 1],
    },
)