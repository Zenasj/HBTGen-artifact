import torch

class DetectionWarmupScheduler(LambdaLR):
    r"""Warm-up learning rate scheduler for detection.

    For iteration x, the learning rate variation is given by:

      /
     |  => (warmup_factor * (1 - alpha) + alpha) if x < warmup_iters
    <
     |  => 1 otherwise
      \

        with alpha = x / warmup_iters.

        For this particular scheduler, warmup_factor = 1/warmup_iters, and
        warmup_iters can't be greater than the dataloader length.

    Args:
        optimizer (torch optimizer): optimizer object.
        warmup_iters (int): number of warmup iterations.
        ndataloader (int): length of the dataloader.

    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_iters: int,
        ndataloader: int,
    ) -> None:
        warmup_factor: float = 1. / warmup_iters
        min_warmup_iters: int = min(warmup_iters, ndataloader - 1)

        def f(x: int) -> float:
            if x >= min_warmup_iters:
                return 1
            alpha: float = float(x) / min_warmup_iters
            return warmup_factor * (1 - alpha) + alpha

        super().__init__(optimizer, lr_lambda=f)