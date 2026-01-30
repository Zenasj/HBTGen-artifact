import torch

def test_unary_functions(self):
        for op in (abs, operator.abs):
            with self.subTest(op=op):
                print("iteration %s" % op)
                def fn(x, y):
                    return x * op(y)

                arg = torch.ones(4)
                opt_fn = torch._dynamo.optimize(nopython=True)(fn)
                print("tensor*abs(constant)")

                # This fail on second iteration.
                # tensor([-2., -2., -2., -2.]) VS tensor([2., 2., 2., 2.])
                print(opt_fn(arg, -2), fn(arg, -2))

                self.assertEqual(opt_fn(arg, -2), fn(arg, -2))

                print("constant*abs(constant)")

                # if we remove this the tests pass!
                self.assertEqual(opt_fn(-2, -3), fn(-2, -3))

def test_unary_functions(self):
        for op in (operator.abs, ):
                print("iteration %s" % op)
                def fn(x, y):
                    return x * op(y)

                arg = torch.ones(4)*4
                opt_fn = torch._dynamo.optimize(nopython=True, backend="inductor", dynamic=True)(fn)
                print("tensor*abs(constant)")

                # This fail on second iteration.
                # tensor([-2., -2., -2., -2.]) VS tensor([2., 2., 2., 2.])
                # print(opt_fn(arg, -2), fn(arg, -2))

                self.assertEqual(opt_fn(arg, -2), fn(arg, -2))

def test_unary_functions(self):
        def fn(x, y):
            return x * abs(y)

        arg = torch.ones(4, device="cuda")*4
        opt_fn = torch.compile(fullgraph=True, backend="inductor", dynamic=True)(fn)

        # This fail on second iteration.
        # tensor([-2., -2., -2., -2.]) VS tensor([2., 2., 2., 2.])
        # print(opt_fn(arg, -2), fn(arg, -2))

        self.assertEqual(opt_fn(arg, -2), fn(arg, -2))

def test_unary_functions(self):
        def fn(x, y):
            return x * abs(y)

        arg = torch.ones(4, device="cuda")*4
        opt_fn = torch.compile(fullgraph=True, backend="inductor", dynamic=True)(fn)
        self.assertEqual(opt_fn(arg, -2), fn(arg, -2))