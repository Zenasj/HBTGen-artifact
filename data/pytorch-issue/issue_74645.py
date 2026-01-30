def my_add(x, y):
  result = x.clone()
  result.add_(y)
  return result