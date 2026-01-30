import torch

def test_unary_functions(self):
        for op in (operator.pos,):
            with self.subTest(op=op):
                def fn(x, y):
                    return x * op(y)

                opt_fn = torch.compile(fullgraph=True, dynamic=True)(fn)
                tensor1 = torch.ones(4)
                tensor2 = torch.ones(4)

                def test(arg1, arg2):
                    self.assertEqual(opt_fn(arg1, arg2), fn(arg1, arg2))

                # test(tensor1, tensor2)
                # print(fn(tensor1, -2))
                # print(opt_fn(tensor1, -2))
                # test(tensor1, -2)
                # test(-2, tensor1)
                test(-2, -2)