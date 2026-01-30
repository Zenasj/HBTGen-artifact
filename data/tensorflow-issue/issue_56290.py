class A:
    def __init__(self, name):
        self._name = name
    def __del__(self):
        print(f"A obj {self._name} is delete")


def _run_rpc():
    a = A("a")
    import tensorflow
    b = A("b")


if __name__ == "__main__":
    _run_rpc()
    print("another start.")