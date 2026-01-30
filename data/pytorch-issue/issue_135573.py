import torch
import enum

class SampleEnum(enum.Enum):
    A = 0
    B = 1

class Sample:
    _enum: SampleEnum

    def is_a(self) -> bool:
        return self._enum == SampleEnum.A

    def is_b(self) -> bool:
        return self._enum == SampleEnum.B

class ASample(Sample):
    _enum = SampleEnum.A

current_sample = ASample()

def foo():
    is_a = torch.tensor(0)
    if current_sample.is_a():
        is_a = torch.tensor(1)
    return is_a

print(torch.compile(foo, fullgraph=True)())