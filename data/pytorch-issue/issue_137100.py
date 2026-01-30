def f(x):
  if x.size(0) >= 10:
    return x + x
  else:
   return x * 3

# zs0 is size oblivious, hint = 1
b1 = guard_size_oblivious(zs0 == 1)
b2 = True if zs0 == 1 else False