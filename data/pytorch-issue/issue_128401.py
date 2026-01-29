# torch.rand(B, 2, dtype=torch.float64)
import torch
class MyModel(torch.nn.Module):
    def forward(self, x):
        a = x[0]
        b = x[1]
        # Operation1: a / b (float64 division)
        op1 = a / b
        op1_tensor = op1.to(torch.float32)
        
        # Operation2: torch.tensor(a)/b → a as float32 divided by b (float64)
        a_float32 = a.to(torch.float32)
        op2 = a_float32 / b  # b is cast to float32, result is float32
        
        # Operation3: a (float64) divided by torch.tensor(b) (float32)
        b_float32 = b.to(torch.float32)
        op3 = a / b_float32  # result is float64
        
        # Operation4: a divided by (b converted to float32 then to float64)
        op4 = a / b_float32.to(torch.float64)  # same as op3, but converted to float32
        op4_tensor = op4.to(torch.float32)
        
        # Operation5: same as op1
        op5 = a / b
        op5_tensor = op5.to(torch.float32)
        
        return (op1_tensor, op2, op3, op4_tensor, op5_tensor)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, dtype=torch.float64) * 1e-38

