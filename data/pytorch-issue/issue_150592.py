import torch

py
from transformers import DistilBertTokenizer, DistilBertModel


tokenizer = DistilBertTokenizer.from_pretrained(
    pretrained_model_name_or_path="distilbert/distilbert-base-uncased",
)

model = DistilBertModel.from_pretrained(
    pretrained_model_name_or_path="distilbert/distilbert-base-uncased",
    device_map="cuda"
)
model.eval()

from torch.export import Dim

vocab_size = tokenizer.vocab_size  # Get the actual vocab size

# Step 1: Define Input
batch_size = 4
dummy_input_ids = torch.randint(0, vocab_size, (batch_size, 128))  # Batch size 2, sequence length 128
dummy_attention_mask = torch.ones((batch_size, 128), dtype=torch.int64)

# Step 2: Define Dynamic shapes
dynamic_shapes = {
    "input_ids": (Dim.DYNAMIC, Dim.DYNAMIC),
    "attention_mask": (Dim.DYNAMIC, Dim.DYNAMIC),
}

# Step 3: Define outputh path
output_path = "distilbert-onnx/model-onnx.onnx"

# Step 4: Export to ONNX
torch.onnx.export(
    model,                                      # PyTorch model
    (dummy_input_ids, dummy_attention_mask),
    output_path,                                # Output file
    export_params=True,                         # Store the trained weights
    opset_version=17,                           # ONNX opset version
    do_constant_folding=True,
    input_names=['input_ids', 'attention_mask'], # Input names
    output_names=['last_hidden_state'], # Output names
    dynamic_shapes=dynamic_shapes,
    dynamo=True,
    verbose=True                               # Detailed output
)

print(f"Model exported to {output_path}")

import onnx
onnx_model = onnx.load(output_path)
print("ONNX Opset Version:", onnx_model.opset_import[0].version)

onnx_model.opset_import