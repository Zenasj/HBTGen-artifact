import torch

def test_ParameterList(self):

        @torch.compile(backend="eager")
        def func():
            def make_param():
                return Parameter(torch.randn(2, 2))

            parameters = []
            # without the following print line we generate:
#             [Parameter containing:
#              tensor([[ 0.0461,  0.4024],
#              [-1.0115,  0.2167]], requires_grad=True)]
                    #
            # with it we geenrate 
# [tensor([[ 0.0461,  0.4024],
#         [-1.0115,  0.2167]], grad_fn=<TracableCreateParameterBackward>)]
            print(parameters)


            parameters.append(make_param())
            print(parameters)