@implements(aten.t.default)
def _(func, types, args, kwargs):
    tensor = args[0]
    print("before transpose, ", tensor.shape)
    shape = tensor.shape[::-1]
    new = tensor.__class__(tensor.layout_tensor.t(), shape, tensor.dtype)
    print("after transpose:", new.shape)
    return return_and_correct_aliasing(func, args, kwargs, new)

