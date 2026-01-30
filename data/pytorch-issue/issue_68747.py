import torch

script_model = torch.jit.load(cfg.model_file, map_location="cpu")
preprocessed_model = cpplib.preprocess(script_model._c)