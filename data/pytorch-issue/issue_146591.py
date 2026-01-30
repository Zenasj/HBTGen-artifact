import torch.nn as nn

# in torch/onnx/utils.py
if GLOBALS.onnx_shape_inference:
    _C._jit_pass_onnx_graph_shape_type_inference(
        graph, params_dict, GLOBALS.export_onnx_opset_version
    )

import torch
from diffusers import SanaPipeline
from torch import nn

# 1. Load the model
pipe = SanaPipeline.from_pretrained(
    "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
    variant="bf16",
    torch_dtype=torch.bfloat16,
).to("cuda")

text_encoder = pipe.text_encoder.eval().to("cuda")
tokenizer = pipe.tokenizer

prompt_text = ["star war, 8K, good"]
tokens = tokenizer(
    prompt_text,
    max_length=77,
    padding="max_length",
    truncation=True,
    return_tensors="pt",
).input_ids.to("cuda")

# 2. Wrap the encoder
class TextEncoderWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids):
        outputs = self.encoder(input_ids, return_dict=False)
        # Return last_hidden_state
        return outputs[0]

wrapped_encoder = TextEncoderWrapper(text_encoder).eval().to("cuda")

# 3. Export to ONNX
onnx_path = "/path/to/text_encoder.onnx"

torch.onnx.export(
    wrapped_encoder,
    (tokens,),
    onnx_path,
    do_constant_folding=True,
    input_names=["input_ids"],
    output_names=["last_hidden_state"],
    # opset_version=...
    # use_external_data_format=True,  # tried enabling this as well
)