class A(IterableDataset):
    def some_method(self):
        print("It's A")


class B(IterableDataset):
    def some_method(self):
        print("It's B")


def test_types():
    a = A()
    b = B()

    # correct isinstance
    assert isinstance(a, A)
    assert isinstance(b, B)

    # incorrect isinstance
    assert isinstance(b, A)
    assert isinstance(a, B)