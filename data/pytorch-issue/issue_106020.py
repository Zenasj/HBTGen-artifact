assert weight is None or weight.dim() == 1
AssertionError

f"MultiMarginLoss: expected weight to be None or 1D tensor, got {weight.dim()}D instead"