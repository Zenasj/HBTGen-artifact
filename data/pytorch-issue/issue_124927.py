def test_tensor_is_complex(x):
        if x.is_complex():
            return x + 1
        else:
            return x - 1