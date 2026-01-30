def test_logger_example(self):
        self.logger.print('hello world')
        for i in range(5):
            self.logger.print('i =', i)
            x = 10 / (3-i)

self.assertEqual(..., msg=f"Sample {sample.name} caused the failure.")

def test_logger_example():
    print('hello world')
    for i in range(5):
        print('i =', i)
        x = 10 / (3 - i)

if input.grad is None:
    raise AssertionError("Input has no grad")

self.assertEqual(input.grad, expected_grad, msg="Gradients mismatch")