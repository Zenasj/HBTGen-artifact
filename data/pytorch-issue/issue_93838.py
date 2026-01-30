import torch
from typing import Dict, List, Tuple
device = torch.device("cuda")
model = torch.load("model.pt")
input_dict: Dict[str, torch.Tensor] ={'input1': torch.tensor([1],dtype=torch.float64,device=device),
'input2': torch.tensor([1],dtype=torch.float64,device=device),
'input3': torch.tensor([1],dtype=torch.float64,device=device)
}
model(input_dict)

torch.onnx.export(user_model, (input_dict, {}), "onnx_model.onnx", verbose = True, input_names = ["input1","input2","input3"], output_names = ["output_spec"])