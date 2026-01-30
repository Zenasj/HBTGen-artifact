class MyClass(object):
   def __init__(self: "MyClass", x: int):    # <--- use string literal of class name
      self.x = x

class SecondClass(object):
    def __init__(self: SecondClass, x: int):
       self.x = x 

y = SecondClass(2)
print("Second: ", y.x)