def normalize(x, eps=None):
    return (x - x.mean()) / x.std().clip(min=eps)