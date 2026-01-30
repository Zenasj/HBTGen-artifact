import torch

def unique_consecutive(g, self, return_inverse, return_counts, dim):
    ...
   # Implementation by calling g.op("Unique", ...)

# Register custom symbolic function
torch.onnx.register_custom_op_symbolic(
    symbolic_name="aten::unique_consecutive",
    symbolic_fn=unique_consecutive,
    opset_version=14,
)