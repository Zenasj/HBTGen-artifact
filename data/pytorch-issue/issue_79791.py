import torch

traced_model = torch.jit.trace(model, example_tokens, strict=False)
optimized_model = optimize_for_mobile(traced_model)
optimized_model._save_for_lite_interpreter("qa_model.ptl")