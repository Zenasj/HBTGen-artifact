# New code
import numpy as np

class TnpArray:
    def __init__(self, data):
        self.data = np.array(data)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return TnpArray(self.data + other)
        else:
            raise TypeError(f"Unsupported operand type(s) for +: 'TnpArray' and '{type(other).__name__}'")

    def __repr__(self):
        return repr(self.data)

# Example usage:
a = TnpArray([1, 2, 3])
result = a + 1.5
print(result)  # Output: array([2.5, 3.5, 4.5])