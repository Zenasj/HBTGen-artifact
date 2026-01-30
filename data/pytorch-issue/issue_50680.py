class A:
    def __init__(self, x):
        if x:
            self._attr = 1

    @property
    def val(self):
        return getattr(self, '_attr')

a = A(False)
print('val' in dir(a))
print(hasattr(a, 'val'))

b = A(True)
print('val' in dir(b))
print(hasattr(b, 'val'))