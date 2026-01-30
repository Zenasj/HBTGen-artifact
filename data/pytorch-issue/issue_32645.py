class LambdaLR(_LRScheduler):
    def __init__(self, optimizer: Optimizer, lr_lambda: float, last_epoch: int=...) -> None: ...

class LambdaLR(_LRScheduler):
    """Sets the learning rate of each parameter group to the initial lr
    times a given function. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        lr_lambda (function or list): A function which computes a multiplicative
            factor given an integer parameter epoch, or a list of such
            functions, one for each group in optimizer.param_groups.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # Assuming optimizer has two groups.
        >>> lambda1 = lambda epoch: epoch // 30
        >>> lambda2 = lambda epoch: 0.95 ** epoch
        >>> scheduler = LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

LRLambdaType = Callable[[int], float]

class LambdaLR(_LRScheduler):
    def __init__(self, optimizer: Optimizer, lr_lambda: Union[LRLambdaType, List[LRLambdaType]], last_epoch: int=...) -> None: ...

LRLambdaType = Callable[[int], float]

class LambdaLR(_LRScheduler):
    def __init__(self, optimizer: Optimizer, lr_lambda: Union[LRLambdaType, List[float]], last_epoch: int=...) -> None: ...