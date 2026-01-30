import onnx
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m")
example_input = {
    "input_ids": torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]),
    "use_cache": True,
    "return_dict": False,
}

output = model(**example_input) 
print(output) # the output is a Tuple[Tensor, Tuple[Tuple[Tensor, Tensor], ...], ...]

torch.onnx.export(
    model,
    example_input,
    "bloom-560m.onnx",
)

onnx_model = onnx.load_model("bloom-560m.onnx")
print(onnx_model.graph.output) # only a single output, the rest of the outputs (cache) is ignored