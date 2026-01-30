print(a.__dict__.get("__getitem__", None))
print(a.__class__.__dict__.get("__getitem__", None))
print(a.__class__.__bases__[0].__dict__.get("__getitem__", None))