py
# test.py
from enum import IntFlag
class Test92319(IntFlag):
    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3
 
values = (Test92319(v) for v in [0, 1, 2, 3])
assert all(value in Test92319 for value in values)  # we can use the Python 'in' operator to check membership of the values in the enum
assert Test92319(4) not in Test92319  # this value should _not_ be found within the members of the enum