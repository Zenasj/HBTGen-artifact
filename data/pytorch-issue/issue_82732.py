class Foo:

    def __init__(self):
        self.data = [0, 1]

    def __getitem__(self, index):
        return self.data[index]

f = Foo()
for i in f:
    print(i)  # 0, 1, no exception is raised