def f(x):
  def g(y):
      return y
  print(id(g.__code__))
  return None

f(1)
f(1)
f(1)