import torch.nn as nn

py
import torch
import numpy as np
import onnxruntime as ort

class UpdateModel(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.params = torch.zeros((2, 1, 10))
        
    def forward(self, update: torch.Tensor, index: torch.LongTensor):
        indices = torch.arange(update.shape[0])
        middle = torch.zeros((1,), dtype=torch.long)
        self.params[index, middle, indices] = update.transpose(1, 0)
        
        return self.params
    
model = UpdateModel()

n = 6

update = torch.ones((n, 1))
kv_index = torch.tensor([0])

model(update, kv_index)

dynamic_axes = {
    "update": {0: "n"},
}

torch.onnx.export(
    model,
    (update, kv_index),
    "./test.onnx",
    input_names=["update", "kv_index"],
    output_names=["updated"],
    dynamic_axes=dynamic_axes,
    opset_version=13,
    verbose=True
)

model_path = "./test.onnx"
# model_path = "./test1.onnx" # Works after manually removing the node
providers = [("CPUExecutionProvider")]
sess_options = ort.SessionOptions()
session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)

def gen_numpy_inputs(n: int, idx: int):
    return {
        "update": 5*np.ones((n, 1), dtype=np.float32),
        "kv_index": np.array([idx], dtype=np.int64)
    }

print(session.run(["updated"], gen_numpy_inputs(n, 0))) # Runs
print(session.run(["updated"], gen_numpy_inputs(1, 0))) # Crashes

py
import onnx
onnx_model = onnx.load(model_path)
graph = onnx_model.graph

for node in graph.node:
    if node.name == "/Reshape":
        node_inp = node.input
        node_out = node.output
        break

new_nodes = [node for node in graph.node if node.name != "/Reshape"]

# Update other nodes to reconnect them
for node in new_nodes:
    for i, input_name in enumerate(node.input):
        if input_name in node_out:
            node.input[i] = node_inp[0]

graph.ClearField('node')
graph.node.extend(new_nodes)
onnx.save(onnx_model, "test1.onnx")