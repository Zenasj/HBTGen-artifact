a = tensor.max(dim=0)
if isinstance(a, tuple):
    print("I am a tuple")