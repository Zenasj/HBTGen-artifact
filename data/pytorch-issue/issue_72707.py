import torch

with open("convmix.pt", "rb") as f:
    state_dict = torch.load(f, map_location=torch.device("cpu"))
state_dict = {k.lstrip("module."): v for k, v in state_dict.items()}
model = ConvMixer(args.hdim, args.depth)
model.load_state_dict(state_dict)  # Argument of type "dict[Unknown, Unknown]" cannot be assigned to parameter "state_dict" of type "OrderedDict[str, Tensor]" in function "load_state_dict"

x : dict[str, torch.Tensor] = {}
model.load_state_dict(state_dict)  # Argument of type "dict[str, Tensor]" cannot be assigned to parameter "state_dict" of type "OrderedDict[str, Tensor]" in function "load_state_dict"