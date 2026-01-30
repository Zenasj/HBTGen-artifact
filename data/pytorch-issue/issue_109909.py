import pickle
from pathlib import Path

import torch
from nnsmith.materialize import Model, Oracle
from torch.onnx.verification import find_mismatch

path = Path("./model_78_3459169476")

# Get the paths for pickles and weights
gir_path: Path = path / "gir.pkl"
oracle_path: Path = path / "oracle.pkl"
weights_path: Path = path / "model.pth"

# Load the model from pickle
with gir_path.open("rb") as f:
    gir = pickle.load(f)
model_type = Model.init("torch", "cpu")
model = model_type.from_gir(gir)

# Load weights from weight path.
model.torch_model.load_state_dict(torch.load(weights_path), strict=False)

# Load oracle
oracle = Oracle.load(oracle_path)

model_args = tuple([torch.from_numpy(val) for key, val in oracle.input.items()])

print(f"Testing: {str(path)}")
graph_info = find_mismatch(
    model.torch_model,
    model_args,
    opset_version=16,
    keep_initializers_as_inputs=False,
)

repro_path = path / "model_repro"
graph_info.export_repro(repro_path)