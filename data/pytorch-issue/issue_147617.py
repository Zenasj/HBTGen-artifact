import logging
import onnxruntime
import numpy as np
import torch
from torch_geometric.nn import GAT

logger = logging.getLogger(__name__)

logger.info("Prepare model")
num_features = 23
num_classes = 12
torch_path = "model.txt"
onnx_path = "model.onnx"
model = GAT(in_channels=num_features, out_channels=num_classes, heads=4,
           hidden_channels=16, num_layers=1, v2=True, dropout=0.0)
best_model_ckpt = torch.load(torch_path, weights_only=False)
model.load_state_dict(best_model_ckpt)
model.eval()
device = torch.device("cpu")
model = model.to(device)

logger.info("Generating dummy data for ONNX exporter")
num_segments = 30
x = torch.randn(num_segments, num_features).to(device)
edge_index = torch.randint(num_segments, size=(2, 58)).to(device)

logger.info("Running torch model on dummy data")
with torch.no_grad():
    result_torch = model(x, edge_index).numpy()

logger.info("Exporting")
opset_version = 16
dynamic_axes = {'x': {0: 'dynamic_input_features'}, 'edge_index': {1: 'dynamic_input_edge_connection'},
                'output': {0: 'dynamic_output_segment'}}
torch.onnx.export(model,
                  (x, edge_index),
                  onnx_path,
                  verify=True,
                  output_names=['output'],
                  input_names=['x', 'edge_index'],
                  dynamic_axes=dynamic_axes,
                  opset_version=opset_version,
                  dynamo=True,
                  report=True
                  )

logger.info("Running ONNX inference")
ort_session = onnxruntime.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
inputs = {'x': x.numpy(), 'edge_index': edge_index.numpy()}
result = ort_session.run(['output'], inputs)
result_onnx = torch.Tensor(result[0]).numpy()
diff = result_torch - result_onnx
logger.warning(f"Results difference: {diff}")
logger.warning(f"Max, Min: {np.max(diff)}, {np.min(diff)}")