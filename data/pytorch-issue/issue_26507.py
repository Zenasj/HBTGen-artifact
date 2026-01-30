def __op__(self, other):
    try:
        return self.op(other)
    except TypeError:
        return NotImplemented