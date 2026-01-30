import torch

class AAA:
    class DUMMY:
        class DUMMY2:
            pass
    def dummy(self):
        def dummy2():
            pass
    class BBB:
        @staticmethod
        def CCC():
            class DDD:
                if True:
                    @staticmethod
                    def EEE():
                        x = [torch.ones(3, 3) for _ in range(5)]
                        return x
            return DDD

def fn():
    return AAA.BBB.CCC().EEE()

opt_fn = torch.compile(fn, backend="eager")

opt_fn()