import inspect

def test_me():
    global Barr
    class Barr:
        pass
    print(inspect.getsource(Barr))

if __name__ == "__main__":
    test_me()