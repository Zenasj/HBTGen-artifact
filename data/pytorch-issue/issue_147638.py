def fn(x):
    ix = x + 1
    a = ix.transpose(0, 1)
    return a.detach(), a