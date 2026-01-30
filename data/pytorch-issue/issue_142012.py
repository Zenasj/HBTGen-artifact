import inspect

def main():
    class MyClass:
        def b(self, x):
            return x

        def a(self, x):
            return x

    src_loc = inspect.getsourcelines(MyClass)[1]


    class MyClass:
        def a(self, x):
            return x + x

    src_loc2 = inspect.getsourcelines(MyClass)[1]

    assert src_loc == src_loc2, f"Expect {src_loc} == {src_loc2}"

main()