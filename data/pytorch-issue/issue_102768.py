max_pool1d: Callable
max_pool2d: Callable
max_pool3d: Callable
...
logsigmoid: Callable
softplus: Callable
softshrink: Callable
...

func: Callable[[Args], Return]

def func(args: Args) -> Return: ...