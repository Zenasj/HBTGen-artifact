# torch.rand(5,5, dtype=torch.float32)
import torch
import torch.nn as nn

class PreprocessAndCalculateModel(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        output1, output2 = torch.lobpcg(x)
        return [output1, output2]

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_cpu = PreprocessAndCalculateModel()
        self.model_gpu = PreprocessAndCalculateModel()

    def forward(self, x):
        output_cpu = self.model_cpu(x.to('cpu'))
        output_gpu = self.model_gpu(x.to('cuda'))
        eigen_cpu = output_cpu[0]
        eigen_gpu = output_gpu[0].cpu()
        eigen_diff = torch.abs(eigen_cpu - eigen_gpu).item()
        
        vec_cpu = output_cpu[1]
        vec_gpu = output_gpu[1].cpu()
        vec_diff = torch.min(
            torch.norm(vec_gpu - vec_cpu, p='fro'),
            torch.norm(vec_gpu + vec_cpu, p='fro')
        ).item()
        
        # Return True if discrepancies exceed thresholds (eigen:1e-4, vector:1e-2)
        return torch.tensor(eigen_diff > 1e-4 or vec_diff > 1e-2, dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(5,5, dtype=torch.float32)

