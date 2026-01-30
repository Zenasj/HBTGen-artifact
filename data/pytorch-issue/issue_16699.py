import torch

dummy_input = torch.ones(1, 14, 200, dtype=torch.float)
dummy_mask = torch.ones(1, 14, dtype=torch.float)
torch.onnx.export(model=model._transformer, args=(dummy_input, dummy_mask), f=args.output_pt, verbose=True)