def _ntuple(x, n):
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)
    return tuple(repeat(x, n))

def _single(x):
    return _ntuple(x, 1)
def _pair(x):
    return _ntuple(x, 2)
def _triple(x):
    return _ntuple(x, 3)
def _quadruple(x):
    return _ntuple(x, 4)