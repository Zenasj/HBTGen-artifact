class Foo():
    def __init__(self):
        super().__init__()

print(Foo.__init__.__code__.co_freevars) # ('__class__',)
print(Foo.__init__.__closure__)          # (<cell at 0x1011fb310: type object at 0x10fe185b0>,)