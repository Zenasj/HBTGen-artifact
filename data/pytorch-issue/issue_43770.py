try:
    something_which_raises_TypeError
except TypeError as e:
    raise ValueError("A more user-friendly exception message.") from e

try:
    something_which_raises_TypeError
except TypeError:
    raise ValueError("A more user-friendly exception message.")

d = {"dog": 1}
key = "cat"
try: 
    print(d[key])  # raises KeyError
except KeyError:
    raise RuntimeError("no member called {}".format(key))

try: 
    print(d[key])  # raises KeyError
except KeyError as e:
    raise RuntimeError("no member called {}".format(key)) from e

try: 
    print(d[key])  # raises KeyError
except KeyError:
    raise RuntimeError("no member called {}".format(key)) from None